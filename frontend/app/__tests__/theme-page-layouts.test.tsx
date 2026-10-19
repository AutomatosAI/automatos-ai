/**
 * PRD-261 A-04d — a ThemeSlot seam on every other page of Gerard's Q3 list (plan §2.6, D0):
 * agents (/agents), missions (/assignments and /missions/[id]), knowledge (/documents),
 * playbooks (/playbooks), the Command Center (/command-center) and home (/).
 *
 * Each route's layout wraps the core page in a ThemeSlot page entry with its keepCorePage list
 * (the app/chat/layout.tsx pattern). With no theme, under builtin Play (which registers none of
 * these pages) and under an installed theme that registers none, the layout renders its children
 * unchanged, the same DOM. An installed theme's registered page draws instead, full bleed inside
 * MainLayout (its frame), unless the link asks for a surface only the core page has. Every route
 * the host lets a theme register (THEME_PAGE_ROUTES) has its seam.
 *
 * Home: the core has no page there. Under Classic, Studio and builtin Play the server forwards to
 * /chat as before; under an installed theme the page renders ThemeSlot, the theme's `/` page
 * draws when it registered one, and otherwise the browser forwards to /chat.
 */
import fs from 'fs'
import path from 'path'
import React, { type ComponentType, type ReactNode } from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import ts from 'typescript'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types'

const state = vi.hoisted(() => ({
  entry: null as ThemeEntry | null,
  layout: 'fallback' as ThemeLayout,
  pathname: '/agents',
  search: '',
  style: undefined as string | undefined,
  replace: vi.fn(),
  redirect: vi.fn((href: string): never => {
    throw new Error(`redirect ${href}`)
  }),
}))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useSearchParams: () => new URLSearchParams(state.search),
  useRouter: () => ({ replace: state.replace }),
  redirect: state.redirect,
}))
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

import AgentsLayout from '@/app/agents/layout'
import AssignmentsLayout from '@/app/assignments/layout'
import CommandCenterLayout from '@/app/command-center/layout'
import DocumentsLayout from '@/app/documents/layout'
import MissionLayout from '@/app/missions/[id]/layout'
import Home from '@/app/page'
import PlaybooksLayout from '@/app/playbooks/layout'
import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { ThemeSlot } from '@/components/theme-host/theme-slot'
import { CoreHomeForward } from '@/components/theme-host/theme-slot-home'
import { BUILTIN_PLAY_ENTRY } from '@/lib/theme-host/builtin/play-entry'
import { THEME_PAGE_ROUTES } from '@/lib/theme-host/constants'
import type { ThemeHostStatus } from '@/lib/theme-host/types'

type Layout = ComponentType<{ children: ReactNode }>

/** Each A-04d layout, the route it registers, and an address on that route. */
const LAYOUTS: ReadonlyArray<readonly [string, Layout, string, string]> = [
  ['app/agents/layout.tsx', AgentsLayout, '/agents', '/agents'],
  ['app/assignments/layout.tsx', AssignmentsLayout, '/assignments', '/assignments'],
  ['app/missions/[id]/layout.tsx', MissionLayout, '/missions/[id]', '/missions/m-1'],
  ['app/documents/layout.tsx', DocumentsLayout, '/documents', '/documents'],
  ['app/playbooks/layout.tsx', PlaybooksLayout, '/playbooks', '/playbooks'],
  ['app/command-center/layout.tsx', CommandCenterLayout, '/command-center', '/command-center'],
]

const INSTALLED_STYLE = 'theme:install:00000000-0000-4000-8000-0000000a04d0'
const INSTALLED_ENTRY: ThemeEntry = {
  ...BUILTIN_PLAY_ENTRY,
  ref: 'install:00000000-0000-4000-8000-0000000a04d0',
  kind: 'installed',
  ownsPages: [],
}
const FRONTEND = path.resolve(__dirname, '..', '..')
const HELLO_A04D = path.resolve(FRONTEND, '..', 'e2e', 'browser', 'fixtures', 'hello-a04d')
/** The fixture's PAGE_ROUTES array, up to its closing bracket on a line of its own. */
const FIXTURE_ROUTES = /const PAGE_ROUTES = \[([\s\S]*?)\n\]/

function HelloFrame({ children }: { children?: React.ReactNode }) {
  return <>{children}</>
}

function ThemePage() {
  return <div data-testid="theme">The theme's page</div>
}

function CorePage() {
  return (
    <main data-testid="core">
      <h1>The core page</h1>
    </main>
  )
}

/** The theme host as the provider gives it for an installed theme that registered `pages`. */
function installedHost(pages: readonly string[], status: ThemeHostStatus = 'active'): ThemeHostContextValue {
  const active = status === 'active'
  return {
    status,
    reason: status === 'failed' ? 'load' : null,
    frame: active ? HelloFrame : null,
    slots: active ? new Map() : null,
    pages: new Map<string, unknown>(active ? pages.map((page) => [page, ThemePage]) : []),
    routes: new Map(),
    reportRenderFailure: vi.fn(),
  } as ThemeHostContextValue
}

function underHost(tree: React.ReactElement, host: ThemeHostContextValue | null) {
  return render(host === null ? tree : <ThemeHostContext.Provider value={host}>{tree}</ThemeHostContext.Provider>)
}

/** The layout around the core page, under `host`; returns the rendered HTML. */
function renderLayout(Layout: Layout, host: ThemeHostContextValue | null = null): string {
  return underHost(
    <Layout>
      <CorePage />
    </Layout>,
    host,
  ).container.innerHTML
}

/** The core page's own HTML, as it renders without any layout. */
function coreHtml(): string {
  const html = render(<CorePage />).container.innerHTML
  cleanup()
  return html
}

function underInstalledTheme(address: string) {
  state.entry = INSTALLED_ENTRY
  state.layout = 'theme'
  state.pathname = address
}

afterEach(() => {
  cleanup()
  state.replace.mockClear()
  state.redirect.mockClear()
  Object.assign(state, { entry: null, layout: 'fallback', pathname: '/agents', search: '', style: undefined })
})

describe('every route a theme may register a page on has its ThemeSlot seam', () => {
  it.each(THEME_PAGE_ROUTES.map((route) => [route]))('%s', (route) => {
    const seam = route === '/' ? 'app/page.tsx' : `app${route}/layout.tsx`
    const source = fs.readFileSync(path.join(FRONTEND, seam), 'utf8')
    expect(source, seam).toMatch(/<ThemeSlot page=\{[A-Z_]+\}/)
  })

  it('lists the six A-04d layouts', () => {
    expect(LAYOUTS.map(([seam]) => fs.existsSync(path.join(FRONTEND, seam)))).toEqual(LAYOUTS.map(() => true))
  })
})

describe.each(LAYOUTS)('%s', (_seam, Layout, route, address) => {
  it('renders its children unchanged with no theme', () => {
    const core = coreHtml()
    state.pathname = address
    expect(renderLayout(Layout)).toBe(core)
  })

  it('renders its children unchanged under builtin Play, which registers no page here', () => {
    const core = coreHtml()
    state.entry = BUILTIN_PLAY_ENTRY
    state.layout = 'theme'
    state.pathname = address
    expect(renderLayout(Layout)).toBe(core)
  })

  it('renders its children unchanged under an installed theme that registered no page here', () => {
    const core = coreHtml()
    underInstalledTheme(address)
    expect(renderLayout(Layout, installedHost(['/chat']))).toBe(core)
  })

  it("renders an installed theme's registered page instead, full bleed inside MainLayout (the theme's frame)", () => {
    underInstalledTheme(address)
    renderLayout(Layout, installedHost([route]))
    expect(screen.getByTestId('main-layout')).toContainElement(screen.getByTestId('theme'))
    expect(screen.getByTestId('main-layout')).toHaveAttribute('data-full-bleed', 'true')
    expect(screen.queryByTestId('core')).toBeNull()
  })

  it('renders its children below the theme width, and nothing while the installed theme loads', () => {
    underInstalledTheme(address)
    state.layout = 'fallback'
    renderLayout(Layout, installedHost([route]))
    expect(screen.getByTestId('core')).toBeInTheDocument()
    cleanup()
    underInstalledTheme(address)
    renderLayout(Layout, installedHost([route], 'loading'))
    expect(screen.queryByTestId('core')).toBeNull()
    expect(screen.queryByTestId('theme')).toBeNull()
  })
})

/** Per address on a layout's route: a query string, whether the link keeps the core page, the layout, its route. */
const LINKS: ReadonlyArray<readonly [string, string, boolean, Layout, string]> = [
  ['/agents', 'tab=skills', true, AgentsLayout, '/agents'],
  ['/agents', 'agent=a1&panel=reports', true, AgentsLayout, '/agents'],
  ['/agents', 'tab=roster', false, AgentsLayout, '/agents'],
  ['/assignments', 'tab=playbooks', true, AssignmentsLayout, '/assignments'],
  ['/assignments', 'id=p1', true, AssignmentsLayout, '/assignments'],
  ['/assignments', 'tab=missions&state=awaiting_approval', false, AssignmentsLayout, '/assignments'],
  ['/missions/m-1', 'tab=tasks', false, MissionLayout, '/missions/[id]'],
  ['/documents', 'q=notes', false, DocumentsLayout, '/documents'],
  ['/playbooks', 'id=p1', true, PlaybooksLayout, '/playbooks'],
  ['/command-center', 'tab=board', true, CommandCenterLayout, '/command-center'],
  ['/command-center', 'openExecution=e1', true, CommandCenterLayout, '/command-center'],
  ['/command-center', 'tab=summary', false, CommandCenterLayout, '/command-center'],
]

describe('keepCorePage: a link into a surface only the core page has keeps it', () => {
  it.each(LINKS)('%s?%s keeps the core page: %s', (address, search, keepsCore, Layout, route) => {
    underInstalledTheme(address)
    state.search = search
    renderLayout(Layout, installedHost([route]))
    expect(screen.queryByTestId('core') !== null).toBe(keepsCore)
    expect(screen.queryByTestId('theme') !== null).toBe(!keepsCore)
  })
})

describe('app/page.tsx (home)', () => {
  it.each([
    ['no cookie', undefined],
    ['Classic', 'classic'],
    ['Studio', 'studio'],
    ['builtin Play, which owns no home page', 'theme:builtin:play'],
    ['a legacy play cookie', 'play'],
  ])('forwards to /chat on the server under %s, as before', async (_label, style) => {
    state.style = style
    await expect(Home()).rejects.toThrow('redirect /chat')
    expect(state.redirect).toHaveBeenCalledWith('/chat')
  })

  it("renders the home ThemeSlot under an installed theme's style, the core's version the forward to /chat", async () => {
    state.style = INSTALLED_STYLE
    const page = await Home()
    expect(page.type).toBe(ThemeSlot)
    expect(page.props.page).toBe('/')
    expect(page.props.children).toEqual(<CoreHomeForward />)
    expect(state.redirect).not.toHaveBeenCalled()
  })
})

describe('the home ThemeSlot in the browser', () => {
  async function renderHome(host: ThemeHostContextValue) {
    state.style = INSTALLED_STYLE
    underInstalledTheme('/')
    underHost(await Home(), host)
  }

  it("renders the installed theme's `/` page inside MainLayout, and never forwards", async () => {
    await renderHome(installedHost(['/']))
    expect(screen.getByTestId('main-layout')).toContainElement(screen.getByTestId('theme'))
    expect(state.replace).not.toHaveBeenCalled()
  })

  it('forwards to /chat when the theme registered no home page, or failed', async () => {
    await renderHome(installedHost(['/agents']))
    expect(state.replace).toHaveBeenCalledWith('/chat')
    cleanup()
    state.replace.mockClear()
    await renderHome(installedHost(['/'], 'failed'))
    expect(state.replace).toHaveBeenCalledWith('/chat')
  })

  it('neither draws nor forwards while the installed theme loads', async () => {
    await renderHome(installedHost(['/'], 'loading'))
    expect(screen.queryByTestId('theme')).toBeNull()
    expect(state.replace).not.toHaveBeenCalled()
  })
})

describe('the browser lane fixture (e2e/browser/fixtures/hello-a04d)', () => {
  function syntaxErrors(file: string): string[] {
    const output = ts.transpileModule(fs.readFileSync(path.join(HELLO_A04D, file), 'utf8'), {
      reportDiagnostics: true,
      fileName: file,
      compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, allowJs: true },
    })
    return (output.diagnostics ?? []).map((diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'))
  }

  it.each(['entry.js', 'entry.frame-only.js'])('%s parses as an ES module', (file) => {
    expect(syntaxErrors(file)).toEqual([])
  })

  it('registers a page on every route of THEME_PAGE_ROUTES, and the frame-only entry on none', () => {
    const listed = fs.readFileSync(path.join(HELLO_A04D, 'entry.js'), 'utf8').match(FIXTURE_ROUTES)?.[1] ?? ''
    expect(Array.from(listed.matchAll(/'([^']+)'/g), (match) => match[1])).toEqual([...THEME_PAGE_ROUTES])
    expect(fs.readFileSync(path.join(HELLO_A04D, 'entry.frame-only.js'), 'utf8')).not.toContain('registerPage')
  })
})
