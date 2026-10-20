/**
 * PRD-261 A-04b — ThemeSlot's page entry on the server: it adds no Suspense boundary of its own,
 * and the core page is never a boundary's fallback.
 *
 * React's streaming server renderer (React 19.2, as Next 15.5 vendors it) outlines a finished
 * boundary over 500 bytes once the bytes before it and its own pass progressiveChunkSize (12,800):
 * it streams the fallback, then the content again in a hidden <div id="S:…"> that a script swaps
 * in later. With the core page as the fallback of a boundary around the core page, /agents and
 * /chat (about 17 KB each) came down twice, and WebKit showed two Studio shells
 * (react/react#37014). The renderer here is the React vitest runs with (package.json), which never
 * counts a boundary's bytes, so it cannot outline; what it does write is the marker of every
 * boundary (`<!--$-->`), so the HTML of a page under no boundary is exactly the page's own HTML.
 * e2e/browser/specs/default-build.spec.ts checks the real server's HTML for /agents, /chat and
 * /deliverables.
 *
 * Where a theme may draw the page (builtin Play's /chat, an installed theme), ThemeSlot renders
 * nothing on the server, where the width and an installed theme's pages are not known: no HTML at
 * all, not even an empty boundary. At the theme layout it renders the theme's page.
 */
import React from 'react'
import { renderToString } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types'

const state = vi.hoisted(() => ({
  entry: null as ThemeEntry | null,
  layout: 'fallback' as ThemeLayout,
  pathname: '/agents',
  search: '',
}))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useSearchParams: () => new URLSearchParams(state.search),
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

import { BUILTIN_PLAY_ENTRY } from '@/lib/theme-host/builtin/play-entry'
import type { ThemeHostStatus } from '@/lib/theme-host/types'

import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'
import { ThemeSlot } from '../theme-slot'

/** Past this many bytes React's streaming renderer outlines a finished boundary (its progressiveChunkSize). */
const PROGRESSIVE_CHUNK_SIZE = 12_800
/** Rows enough to take the core page well past it, as the Agents page is. */
const CORE_PAGE_ROWS = 400
const SHELL_CLASS = 'class="sh-shell'
/** What a Suspense boundary leaves in the server's HTML: its markers, an outlined one's template and hidden segment. */
const BOUNDARY_MARKS = ['<!--$', '<!--/$', '<template id="B:', 'id="S:']
/** What ThemeSlot sends where a theme may draw and nothing is known yet: no HTML, not even an empty boundary. */
const NOTHING = ''
/** The theme page's text: no character the server escapes, so it reads as written in the HTML. */
const THEME_TEXT = 'The installed theme page'
const INSTALLED_ENTRY: ThemeEntry = {
  ...BUILTIN_PLAY_ENTRY,
  ref: 'install:00000000-0000-4000-8000-0000000a04b2',
  kind: 'installed',
  ownsPages: ['/chat'],
}

/** A core page as large as a real one: the Studio shell around a long list. */
function CorePage() {
  return (
    <div className="sh-shell safe-bottom">
      <ul>
        {Array.from({ length: CORE_PAGE_ROWS }, (_, index) => (
          <li key={index}>{`Agent ${index}: one row of the core page`}</li>
        ))}
      </ul>
    </div>
  )
}

function ThemePage() {
  return <div data-testid="theme">{THEME_TEXT}</div>
}

function HelloFrame({ children }: { children?: React.ReactNode }) {
  return <>{children}</>
}

/** The theme host's value for `status`: Studio's ('off'), or an installed theme's with a page on /chat. */
function hostValue(status: ThemeHostStatus): ThemeHostContextValue {
  const active = status === 'active'
  return {
    status,
    reason: null,
    frame: active ? HelloFrame : null,
    slots: active ? new Map() : null,
    pages: active ? new Map<string, unknown>([['/chat', ThemePage]]) : null,
    routes: active ? new Map() : null,
    reportRenderFailure: vi.fn(),
  }
}

/** The server's HTML for a page entry on `page` around the core page, under `host` when given. */
function serverHtml(page: string, host: ThemeHostContextValue | null = null): string {
  const slot = (
    <ThemeSlot page={page} keepCorePage={['tab']}>
      <CorePage />
    </ThemeSlot>
  )
  return renderToString(host === null ? slot : <ThemeHostContext.Provider value={host}>{slot}</ThemeHostContext.Provider>)
}

function occurrences(html: string, text: string): number {
  return html.split(text).length - 1
}

/** Where no theme may draw /agents on the server: each case sets the state and returns the theme host's value. */
const NO_THEME_PAGE_CASES: ReadonlyArray<readonly [string, () => ThemeHostContextValue | null]> = [
  ['with no theme host (an isolated mount)', () => null],
  ['under Studio, the host off', () => hostValue('off')],
  [
    'on a link into a surface only the core page has',
    () => {
      state.search = 'tab=skills'
      return hostValue('off')
    },
  ],
  [
    'under builtin Play, which registered no page on /agents, the width not known',
    () => {
      state.entry = BUILTIN_PLAY_ENTRY
      state.layout = 'pending'
      return null
    },
  ],
  [
    "on a link the core page keeps, while an installed theme's code loads",
    () => {
      state.search = 'tab=skills'
      return hostValue('loading')
    },
  ],
]

afterEach(() => {
  Object.assign(state, { entry: null, layout: 'fallback', pathname: '/agents', search: '' })
})

describe('ThemeSlot on the server, where no theme may draw the page', () => {
  const corePage = renderToString(<CorePage />)

  it('is given a core page past the size React outlines', () => {
    expect(corePage.length).toBeGreaterThan(PROGRESSIVE_CHUNK_SIZE)
  })

  it.each(NO_THEME_PAGE_CASES)('sends the core page exactly once, in no Suspense boundary, %s', (_label, setUp) => {
    const html = serverHtml('/agents', setUp())
    expect(html).toBe(corePage)
    expect(occurrences(html, SHELL_CLASS)).toBe(1)
    expect(BOUNDARY_MARKS.filter((mark) => html.includes(mark))).toEqual([])
  })
})

describe("ThemeSlot on the server, where a theme may draw the page: nothing yet, or the theme's page", () => {
  it("renders nothing for builtin Play's registered /chat page while the width is not known", () => {
    state.entry = BUILTIN_PLAY_ENTRY
    state.layout = 'pending'
    state.pathname = '/chat'
    expect(serverHtml('/chat')).toBe(NOTHING)
  })

  it("renders nothing while an installed theme's code loads, whose pages are not known yet", () => {
    state.pathname = '/chat'
    expect(serverHtml('/chat', hostValue('loading'))).toBe(NOTHING)
  })

  it("renders the theme's page, never the core page, at the theme layout", () => {
    state.entry = INSTALLED_ENTRY
    state.layout = 'theme'
    state.pathname = '/chat'
    const html = serverHtml('/chat', hostValue('active'))
    expect(html).toContain(THEME_TEXT)
    expect(html).not.toContain(SHELL_CLASS)
  })
})
