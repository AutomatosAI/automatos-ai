/**
 * PRD-260 M90, PRD-261 A-02b — ⌘K finds the active theme's own pages. They come from the theme's
 * registry entry (`nav`), not a list in the hook: under an installed theme, its entry's (here Play's
 * package as the boot route answers it: Conversations, Make it yours at /looks and the style sheet at
 * /looks/style); under Classic and Studio, none, so the search never offers a page that shows only the
 * core's note there; under another theme, that theme's own.
 */
import { describe, it, expect, vi, beforeAll, afterEach } from 'vitest'
import { renderHook, act, render, screen, fireEvent } from '@testing-library/react'

import type { ThemeEntry, ThemeNavItem } from '@/lib/theme-host/registry-types'

const nav = vi.hoisted(() => ({ push: vi.fn(), theme: null as ThemeEntry | null }))

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: nav.push, replace: vi.fn() }) }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => null }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useActiveTheme: () => nav.theme }))

import { apiClient } from '@/lib/api-client'
import { useGlobalSearch } from '@/hooks/use-global-search'
import { GlobalSearch } from '@/components/shared/global-search'
import { GLOBAL_SEARCH_OPEN_EVENT } from '@/lib/global-search-event'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY } from '@/lib/theme-host/__tests__/installed-entry-fixture'

function installedTheme(): ThemeEntry {
  const entry = installedThemeEntry(INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY)
  if (entry === null) throw new Error('no installed entry for the Play package')
  return entry
}

/** An installed theme with pages of its own, as the root layout builds its entry from the boot. */
const THEME = installedTheme()
const CONVERSATIONS = 'Conversations'
const MAKE_IT_YOURS = 'Make it yours'
const STYLE_SHEET = 'Style sheet'

/** The path of the theme's nav item `label`. */
function pathOf(label: string): string {
  const item = THEME.nav.find((entry) => entry.label === label)
  if (item === undefined) throw new Error(`the theme lists no page "${label}"`)
  return item.path
}

function pagesFor(text: string, underTheme: boolean): string[] {
  return pagesWithNav(text, underTheme ? THEME.nav : undefined)
}

function pagesWithNav(text: string, themeNav: readonly ThemeNavItem[] | undefined): string[] {
  const { result } = renderHook(() => useGlobalSearch(null, themeNav))
  act(() => result.current.setQuery(text))
  return result.current.pages.map((page) => page.path)
}

afterEach(() => {
  vi.restoreAllMocks()
  nav.push.mockClear()
  nav.theme = null
})

describe("global search · the active theme's pages (hook)", () => {
  it("under an installed theme, 'conv' finds its Conversations; under Studio it doesn't", () => {
    vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
    expect(pagesFor('conv', true)).toEqual([pathOf(CONVERSATIONS)])
    expect(pagesFor('conv', false)).toEqual([])
  })

  it('under an installed theme, finds Make it yours and the style sheet by their names', () => {
    vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
    expect(pagesFor('make it', true)).toEqual([pathOf(MAKE_IT_YOURS)])
    expect(pagesFor('style sheet', true)).toEqual([pathOf(STYLE_SHEET)])
    expect(pagesFor('make it', false)).toEqual([])
    expect(pagesFor('style sheet', false)).toEqual([])
  })

  it('lists them with no query only under the theme, after the core pages', () => {
    const themed = pagesFor('', true)
    const studio = pagesFor('', false)
    const own = [CONVERSATIONS, MAKE_IT_YOURS, STYLE_SHEET].map(pathOf)
    expect(themed.slice(-own.length)).toEqual(own)
    expect(themed.slice(0, -own.length)).toEqual(studio)
    expect(studio).not.toContain(pathOf(CONVERSATIONS))
  })

  it("lists another theme's own pages from its entry, and none of the first theme's", () => {
    vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
    const garden: readonly ThemeNavItem[] = [{ id: 'nav-garden-plots', label: 'Plots', path: '/x/plots' }]
    expect(pagesWithNav('plots', garden)).toEqual(['/x/plots'])
    expect(pagesWithNav('conv', garden)).toEqual([])
    expect(pagesWithNav('', garden).slice(-1)).toEqual(['/x/plots'])
  })
})

describe("global search · the active theme's pages (the dialog reads the style)", () => {
  beforeAll(() => {
    // jsdom has no layout: cmdk's list measures itself, and scrolls its active item into view.
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      unobserve() {}
      disconnect() {}
    })
    if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = () => {}
  })

  async function searchInDialog(text: string) {
    vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
    render(<GlobalSearch />)
    act(() => {
      window.dispatchEvent(new Event(GLOBAL_SEARCH_OPEN_EVENT))
    })
    fireEvent.change(await screen.findByPlaceholderText(/Search pages/), { target: { value: text } })
  }

  it('under an installed theme, typing "conv" offers its Conversations and opens it', async () => {
    nav.theme = THEME
    await searchInDialog('conv')
    fireEvent.click(await screen.findByText(CONVERSATIONS))
    expect(nav.push).toHaveBeenCalledWith(pathOf(CONVERSATIONS))
  })

  it('under Studio, typing "conv" offers no Conversations page', async () => {
    await searchInDialog('conv')
    expect(await screen.findByPlaceholderText(/Search pages/)).toHaveValue('conv')
    expect(screen.queryByText(CONVERSATIONS)).toBeNull()
  })
})
