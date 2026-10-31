/**
 * PRD-261 A-07d, A-02c — a private theme version the member has not opted in to, chosen from the
 * header's theme menu or ⌘K, asks first, as "Use this theme" in Settings does.
 *
 * The bug: the menu and ⌘K stored such a choice straight away; the page reloaded, the host refused
 * to run code the member had not opted in to, and Studio showed instead. Now, from either:
 *
 * - The entry is noted "Private · opt in first"; an opted-in private version keeps "Private · not
 *   reviewed".
 * - Choosing it opens the one opt-in dialog (components/settings/themes/choose-install.tsx) with the
 *   server's wording, and stores no choice.
 * - "Opt in and use" posts the opt-in for the version shown; only once the server recorded it is the
 *   choice stored (the install's id), and the page reloads into the theme.
 * - Cancel chooses nothing: no opt-in, no choice, no reload, the style as it was.
 * - An install that needs no opt-in is chosen at once, as before.
 *
 * Mounted as components/providers.tsx nests them: the style provider, the query client, then the
 * chooser around the header's menu and ⌘K (which mounts ThemeChoiceSync, which fetches the installs
 * once the menu or ⌘K opens). apiClient is the real one with its theme methods stubbed.
 */
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-opt-in' } }),
  useWorkspace: () => ({ workspace: { id: 'ws-opt-in' } }),
}))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn() } }))
// The local step-up: the opt-in route answers at once (lib/theme-host/api.ts useStepUp).
vi.mock('@clerk/nextjs', () => ({
  useReverification: (fetcher: (...args: unknown[]) => Promise<unknown>) => fetcher,
}))

import { ChooseInstallProvider } from '@/components/settings/themes/choose-install'
import { OPT_IN_CONFIRM } from '@/components/settings/themes/opt-in-dialog'
import { INSTALL_ID, OPT_IN_WORDING, PACKAGE_ID, installRow } from '@/components/settings/themes/__tests__/theme-fixtures'
import { GlobalSearch } from '@/components/shared/global-search'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { apiClient, type InstalledTheme } from '@/lib/api-client'
import { GLOBAL_SEARCH_OPEN_EVENT } from '@/lib/global-search-event'
import { forgetChecksInThisDocumentForTests } from '@/lib/theme-host/choice-sync'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE } from '@/lib/ui-style'

const OPT_IN_FIRST = 'Private · opt in first'
const PRIVATE_NOT_REVIEWED = 'Private · not reviewed'
const INSTALLED_STYLE = `theme:install:${INSTALL_ID}`
const THEME_NAME = 'Hello theme'
const DIALOG = 'theme-opt-in-dialog'

/** A private version this member has not opted in to (the fixture's row: optIn.required). */
const ASKS_FIRST = installRow()
/** The same install once this member opted in to its version. */
const OPTED_IN = installRow({ optIn: { required: false, wording: OPT_IN_WORDING, packageId: PACKAGE_ID } })
/** Another private install, opted in to. */
const DUNE = installRow({
  installId: '44444444-4444-4444-8444-444444444444', themeId: 'acme/dune', name: 'Dune',
  optIn: { required: false, wording: OPT_IN_WORDING, packageId: PACKAGE_ID },
})
const OPT_IN_RECORDED = { installId: INSTALL_ID, optIn: { required: false, wording: OPT_IN_WORDING, packageId: PACKAGE_ID } }

function StyleShown() {
  return <span data-testid="style">{useUiStyle().style}</span>
}

/** The header's menu and ⌘K, nested as components/providers.tsx nests them; returns the page's reload. */
function renderApp(installs: readonly InstalledTheme[]) {
  vi.spyOn(apiClient, 'getInstalledThemes').mockResolvedValue({ installs: [...installs] })
  const reload = vi.fn()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <UiStyleProvider initialStyle="studio" reloadPage={reload}>
      <QueryClientProvider client={client}>
        <ChooseInstallProvider>
          <StyleShown />
          <ThemeToggle />
          <GlobalSearch />
        </ChooseInstallProvider>
      </QueryClientProvider>
    </UiStyleProvider>,
  )
  return reload
}

interface Surface {
  readonly name: string
  /** Opens the surface the way a member does; resolves to its menu or dialog. */
  readonly open: () => Promise<HTMLElement>
  /** The entry for the theme named `label`, once the installs are listed. */
  readonly entry: (container: HTMLElement, label: string) => Promise<HTMLElement>
}

const THEME_MENU: Surface = {
  name: 'the theme menu',
  open: async () => {
    const trigger = await screen.findByRole('button', { name: 'Appearance' })
    await waitFor(() => expect(trigger).toBeEnabled())
    fireEvent.keyDown(trigger, { key: 'Enter' }) // Radix opens on Enter, as for a keyboard user
    return screen.findByRole('menu')
  },
  entry: (menu, label) => within(menu).findByRole('menuitemradio', { name: new RegExp(label) }),
}

const COMMAND_K: Surface = {
  name: '⌘K',
  open: async () => {
    act(() => { window.dispatchEvent(new Event(GLOBAL_SEARCH_OPEN_EVENT)) })
    fireEvent.change(await screen.findByPlaceholderText(/Search pages/), { target: { value: 'theme' } })
    return screen.findByRole('dialog')
  },
  entry: (dialog, label) => within(dialog).findByRole('option', { name: new RegExp(label) }),
}

async function pick(surface: Surface, label: string) {
  const item = await surface.entry(await surface.open(), label)
  act(() => { fireEvent.click(item) })
}

/** Lets pending promises and the effects after them run. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

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
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  window.sessionStorage.clear()
  vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
  vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue({ choice: 'studio', source: 'choice', theme: null })
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe.each([THEME_MENU, COMMAND_K])('$name: a private version not opted in to asks first', (surface) => {
  it('is noted "Private · opt in first"; an opted-in private one keeps "Private · not reviewed"', async () => {
    renderApp([ASKS_FIRST, DUNE])
    const container = await surface.open()
    expect(await surface.entry(container, THEME_NAME)).toHaveTextContent(OPT_IN_FIRST)
    const dune = await surface.entry(container, 'Dune')
    expect(dune).toHaveTextContent(PRIVATE_NOT_REVIEWED)
    expect(dune).not.toHaveTextContent(OPT_IN_FIRST)
  })

  it("opens the opt-in dialog with the server's wording, and stores no choice", async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const optIn = vi.spyOn(apiClient, 'optInTheme').mockResolvedValue(OPT_IN_RECORDED)
    const reload = renderApp([ASKS_FIRST])
    await pick(surface, THEME_NAME)

    const dialog = await screen.findByTestId(DIALOG)
    expect(within(dialog).getByTestId('theme-opt-in-wording')).toHaveTextContent(OPT_IN_WORDING)
    await settle()
    expect(put).not.toHaveBeenCalled()
    expect(optIn).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
    expect(document.cookie).not.toContain(INSTALLED_STYLE)
  })

  it('"Opt in and use" opts in to the version shown, and only once the server recorded it stores the choice and reloads', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    let recordOptIn: () => void = () => {}
    const optIn = vi.spyOn(apiClient, 'optInTheme').mockImplementation(
      () => new Promise((resolve) => { recordOptIn = () => resolve(OPT_IN_RECORDED) }),
    )
    const reload = renderApp([ASKS_FIRST])
    await pick(surface, THEME_NAME)
    fireEvent.click(within(await screen.findByTestId(DIALOG)).getByRole('button', { name: OPT_IN_CONFIRM }))

    await waitFor(() => expect(optIn).toHaveBeenCalledWith(INSTALL_ID, PACKAGE_ID))
    await settle()
    expect(put).not.toHaveBeenCalled()
    await act(async () => { recordOptIn() })
    await waitFor(() => expect(put).toHaveBeenCalledWith(INSTALL_ID))
    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1))
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${INSTALLED_STYLE}`)
    await waitFor(() => expect(screen.queryByTestId(DIALOG)).toBeNull())
  })

  it('Cancel chooses nothing: no opt-in, no choice, no reload, the style as it was', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const optIn = vi.spyOn(apiClient, 'optInTheme').mockResolvedValue(OPT_IN_RECORDED)
    const reload = renderApp([ASKS_FIRST])
    await pick(surface, THEME_NAME)
    fireEvent.click(within(await screen.findByTestId(DIALOG)).getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect(screen.queryByTestId(DIALOG)).toBeNull())
    await settle()
    expect(optIn).not.toHaveBeenCalled()
    expect(put).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
    expect(document.cookie).not.toContain(INSTALLED_STYLE)
    expect(screen.getByTestId('style')).toHaveTextContent('studio')
  })

  it('an install that needs no opt-in is chosen at once, as before: stored by its id, then the reload', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const optIn = vi.spyOn(apiClient, 'optInTheme').mockResolvedValue(OPT_IN_RECORDED)
    const reload = renderApp([OPTED_IN])
    await pick(surface, THEME_NAME)

    await waitFor(() => expect(put).toHaveBeenCalledWith(INSTALL_ID))
    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1))
    expect(screen.queryByTestId(DIALOG)).toBeNull()
    expect(optIn).not.toHaveBeenCalled()
  })
})
