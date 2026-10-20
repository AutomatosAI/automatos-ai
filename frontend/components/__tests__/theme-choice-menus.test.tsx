/**
 * PRD-261 A-02c — installed themes in the theme menu and ⌘K, and every pick stored on the server.
 *
 * - An install the workspace has joins both lists as theme:install:<id>, labelled "Private · not
 *   reviewed" when nobody reviewed it; choosing it stores the choice (its id), then reloads.
 * - Choosing builtin Play sends 'builtin:play' and applies it at once.
 * - D1-c (Gerard, 18 Oct): once automatos/play is installed, builtin Play leaves the menu and ⌘K,
 *   except for a member still on it (A-18 moves them).
 *
 * The installs reach the style context the way ThemeChoiceSync hands them over (setInstalledThemes).
 */
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useEffect, type ReactElement, type ReactNode } from 'react'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))
const nav = vi.hoisted(() => ({ push: vi.fn() }))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: nav.push, replace: vi.fn() }) }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-menus' } }),
  useWorkspace: () => ({ workspace: { id: 'ws-menus' } }),
}))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn() } }))
// A plain anchor that forwards its ref, as Radix's asChild item ('Customise look…' under a theme) needs.
vi.mock('next/link', async () => {
  const { forwardRef } = await vi.importActual<typeof import('react')>('react')
  return {
    default: forwardRef<HTMLAnchorElement, { href: string; children: ReactNode }>(function MockLink({ href, children, ...rest }, ref) {
      return <a ref={ref} href={href} {...rest}>{children}</a>
    }),
  }
})

import { GlobalSearch } from '@/components/shared/global-search'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { apiClient, type InstalledTheme } from '@/lib/api-client'
import { BUILTIN_PLAY_HTML_CLASS, BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { forgetChecksInThisDocumentForTests } from '@/lib/theme-host/choice-sync'
import { GLOBAL_SEARCH_OPEN_EVENT } from '@/lib/global-search-event'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, STUDIO_HTML_CLASS, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

const GARDEN_ID = '11111111-1111-4111-8111-111111111111'
const PLAY_PACKAGE_INSTALL_ID = '44444444-4444-4444-8444-444444444444'
const PRIVATE_NOT_REVIEWED = 'Private · not reviewed'

function installRow(overrides: Partial<InstalledTheme>): InstalledTheme {
  return {
    installId: GARDEN_ID, themeId: 'acme/garden', name: 'Garden', packageId: '22222222-2222-4222-8222-222222222222',
    version: '1.0.0', ownerType: 'workspace', reviewed: false, private: true, isDefault: false,
    pinnedVersion: null, previousPackageId: null, pendingPackageId: null, pendingVersion: null,
    installedAt: null, optIn: { required: false, wording: '', packageId: '' }, revoked: null, needsUpdate: false,
    ...overrides,
  }
}

const GARDEN = installRow({})
const INSTALLED_PLAY = installRow({ installId: PLAY_PACKAGE_INSTALL_ID, themeId: 'automatos/play', name: 'Play', reviewed: true })

/** Hands `rows` to the style context, as ThemeChoiceSync does with the workspace's installs. */
function Installs({ rows }: { rows: readonly InstalledTheme[] }) {
  const { setInstalledThemes } = useUiStyle()
  useEffect(() => setInstalledThemes(rows), [rows, setInstalledThemes])
  return null
}

function renderWith(child: ReactElement, rows: readonly InstalledTheme[], style: UiStyle = 'studio', reload = vi.fn()) {
  render(
    <UiStyleProvider initialStyle={style} reloadPage={reload}>
      <Installs rows={rows} />
      {child}
    </UiStyleProvider>,
  )
  return reload
}

/** Opens the Appearance menu the way a keyboard user does (Radix opens on Enter). */
async function openMenu() {
  const trigger = await screen.findByRole('button', { name: 'Appearance' })
  await waitFor(() => expect(trigger).toBeEnabled())
  fireEvent.keyDown(trigger, { key: 'Enter' })
  return screen.findByRole('menu')
}

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

async function searchFor(text: string) {
  act(() => { window.dispatchEvent(new Event(GLOBAL_SEARCH_OPEN_EVENT)) })
  fireEvent.change(await screen.findByPlaceholderText(/Search pages/), { target: { value: text } })
  return screen.findByRole('dialog')
}

const themeItems = (dialog: HTMLElement) => within(within(dialog).getByText('Themes').parentElement as HTMLElement).queryAllByRole('option')

beforeAll(() => {
  // jsdom has no layout: cmdk's list measures itself, and scrolls its active item into view.
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
  if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = () => {}
})

beforeEach(() => {
  forgetChecksInThisDocumentForTests()
  document.documentElement.classList.remove(STUDIO_HTML_CLASS, BUILTIN_PLAY_HTML_CLASS)
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  nav.push.mockClear()
})

describe('the theme menu (A-02c)', () => {
  it('lists an installed theme after the builtin ones, noted when nobody reviewed it', async () => {
    renderWith(<ThemeToggle />, [GARDEN])
    const menu = await openMenu()
    const garden = within(menu).getByRole('menuitemradio', { name: /Garden/ })
    expect(garden).toHaveTextContent(PRIVATE_NOT_REVIEWED)
    const names = within(menu).getAllByRole('menuitemradio').map((item) => item.textContent)
    expect(names.slice(0, 3)).toEqual(['Classic', 'Studio', 'Play'])
  })

  it('choosing an installed theme stores its id, then reloads into it', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const reload = renderWith(<ThemeToggle />, [GARDEN])
    const menu = await openMenu()
    act(() => { fireEvent.click(within(menu).getByRole('menuitemradio', { name: /Garden/ })) })
    await settle()
    expect(put).toHaveBeenCalledWith(GARDEN_ID)
    expect(reload).toHaveBeenCalledTimes(1)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=theme:install:${GARDEN_ID}`)
  })

  it("choosing builtin Play sends 'builtin:play' and applies it at once", async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const reload = renderWith(<ThemeToggle />, [])
    const menu = await openMenu()
    act(() => { fireEvent.click(within(menu).getByRole('menuitemradio', { name: 'Play' })) })
    expect(document.documentElement.classList.contains(BUILTIN_PLAY_HTML_CLASS)).toBe(true)
    await settle()
    expect(put).toHaveBeenCalledWith('builtin:play')
    expect(reload).not.toHaveBeenCalled()
  })

  it('D1-c: with automatos/play installed, the one Play entry is the installed one', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    renderWith(<ThemeToggle />, [INSTALLED_PLAY])
    const menu = await openMenu()
    const plays = within(menu).getAllByRole('menuitemradio', { name: /Play/ })
    expect(plays).toHaveLength(1)
    act(() => { fireEvent.click(plays[0]) })
    await settle()
    expect(put).toHaveBeenCalledWith(PLAY_PACKAGE_INSTALL_ID)
  })

  it('D1-c: a member still on builtin Play keeps its entry, checked and noted "Built in"', async () => {
    renderWith(<ThemeToggle />, [INSTALLED_PLAY], BUILTIN_PLAY_STYLE)
    const menu = await openMenu()
    const builtin = within(menu).getByRole('menuitemradio', { name: /Built in/ })
    expect(builtin).toHaveAttribute('aria-checked', 'true')
    expect(within(menu).getAllByRole('menuitemradio', { name: /Play/ })).toHaveLength(2)
  })
})

describe('⌘K (A-02c)', () => {
  it('finds an installed theme by name, noted when nobody reviewed it, and choosing it stores the choice', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const reload = renderWith(<GlobalSearch />, [GARDEN])
    const dialog = await searchFor('gard')
    const garden = await within(dialog).findByText('Garden')
    expect(within(dialog).getByText(PRIVATE_NOT_REVIEWED)).toBeInTheDocument()
    fireEvent.click(garden)
    await settle()
    expect(put).toHaveBeenCalledWith(GARDEN_ID)
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it("'theme' lists every style: Classic, Studio, builtin Play and the installs", async () => {
    renderWith(<GlobalSearch />, [GARDEN])
    const dialog = await searchFor('theme')
    await within(dialog).findByText('Garden')
    const listed = themeItems(dialog).map((item) => item.textContent)
    expect(listed).toHaveLength(4)
    expect(listed).toEqual(expect.arrayContaining(['Classic', 'Studio', 'Play', `Garden${PRIVATE_NOT_REVIEWED}`]))
  })

  it("choosing builtin Play from ⌘K sends 'builtin:play'", async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    renderWith(<GlobalSearch />, [])
    const dialog = await searchFor('play')
    fireEvent.click(await within(dialog).findByText('Play'))
    await settle()
    expect(put).toHaveBeenCalledWith('builtin:play')
  })

  it('D1-c: with automatos/play installed, ⌘K offers only the installed Play', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    renderWith(<GlobalSearch />, [INSTALLED_PLAY])
    const dialog = await searchFor('play')
    const plays = await within(dialog).findAllByText('Play')
    expect(plays).toHaveLength(1)
    fireEvent.click(plays[0])
    await settle()
    expect(put).toHaveBeenCalledWith(PLAY_PACKAGE_INSTALL_ID)
  })
})
