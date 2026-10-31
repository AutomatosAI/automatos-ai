/**
 * PRD-270 US-270-09b (§5.7, §8-Q8, K22) — back on the Marketplace with
 * `?tab=themes&item=<listing id>&add=1`: the Themes tab picks the listing in the hero and opens its
 * detail with Add to my workspace focused; a member who can't manage themes sees the ask-an-admin
 * sentence; nothing installs until the click; an unlisted item opens nothing; closing strips
 * `item` and `add`, the tab kept.
 *
 * The tab strip hands the link to the tab (marketplace-homepage.tsx); here it is given as the prop,
 * read from the same address by the same `marketplaceLink`. The API doubles are the storefront
 * test's (storefront.test.tsx).
 */
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  getThemeListings: vi.fn(),
  getThemeListing: vi.fn(),
  installThemeListing: vi.fn(),
  getInstalledThemes: vi.fn(),
  getThemeSettings: vi.fn(),
  getActiveTheme: vi.fn(),
  putThemeChoice: vi.fn(),
  importTheme: vi.fn(),
  optInTheme: vi.fn(),
  setDefaultTheme: vi.fn(),
  compareThemeUpdate: vi.fn(),
  updateThemeInstall: vi.fn(),
  rollbackTheme: vi.fn(),
  uninstallTheme: vi.fn(),
  putThemeSettings: vi.fn(),
  submitTheme: vi.fn(),
}))
const workspace = vi.hoisted(() => ({ role: 'owner' as string }))
const nav = vi.hoisted(() => ({ params: new URLSearchParams(), replace: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: workspace.role } }),
}))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => ({ setStyle: vi.fn() }) }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', async () => {
  const { isReverificationHint } = await import('@clerk/shared/authorization-errors')
  type Fetcher = (...args: unknown[]) => Promise<unknown>
  return {
    useReverification: (fetcher: Fetcher) => async (...args: unknown[]) => {
      const result = await fetcher(...args)
      return isReverificationHint(result) ? fetcher(...args) : result
    },
  }
})
vi.mock('next/navigation', () => ({
  useSearchParams: () => nav.params,
  useRouter: () => ({ replace: nav.replace }),
}))

import { installRow, renderWithQuery, themeSettings } from '@/components/settings/themes/__tests__/theme-fixtures'

import { marketplaceLink } from '../../deep-link/deep-link-params'
import { ADD_TO_WORKSPACE, MANAGERS_ADD } from '../listing-format'
import { MarketplaceThemesTab } from '../tab'
import { listing, listingDetail } from './listing-fixtures'

const OWL_ID = 'nightco.owl'
const OWL = { listingId: OWL_ID, themeId: 'nightco/owl', name: 'Night Owl' }

/** The tab as the tab strip renders it for this address. */
function renderLinked(params: Record<string, string>) {
  nav.params = new URLSearchParams(params)
  return renderWithQuery(<MarketplaceThemesTab searchQuery="" link={marketplaceLink(nav.params)} />)
}

async function heroName() {
  return within(await screen.findByTestId('storefront-hero')).getByTestId('storefront-hero-name')
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
  nav.replace.mockReset()
  workspace.role = 'owner'
  // Aurora is first, so the hero would open on it: the link's Night Owl is second.
  api.getThemeListings.mockResolvedValue({ listings: [listing(), listing(OWL)] })
  api.getThemeListing.mockImplementation(async (id: string) => (id === OWL_ID ? listingDetail(OWL) : listingDetail()))
  api.getInstalledThemes.mockResolvedValue({ installs: [] })
  api.getActiveTheme.mockResolvedValue({ choice: 'studio', theme: null })
  api.getThemeSettings.mockResolvedValue(themeSettings())
})

describe('?tab=themes&item=&add=1', () => {
  it('selects the listing in the hero and opens its dialog with Add to my workspace focused', async () => {
    renderLinked({ tab: 'themes', item: OWL_ID, add: '1' })

    const detail = await screen.findByTestId('theme-listing-detail')
    const add = await within(detail).findByRole('button', { name: ADD_TO_WORKSPACE })
    await waitFor(() => expect(add).toHaveFocus())
    expect(api.getThemeListing).toHaveBeenCalledWith(OWL_ID)
    expect(await heroName()).toHaveTextContent('Night Owl')
    expect(api.installThemeListing).not.toHaveBeenCalled()
  })

  it('a member who can’t manage themes sees the ask-an-admin sentence, and no Add', async () => {
    workspace.role = 'member'
    renderLinked({ tab: 'themes', item: OWL_ID, add: '1' })

    const detail = await screen.findByTestId('theme-listing-detail')
    expect(await within(detail).findByText(MANAGERS_ADD)).toBeInTheDocument()
    expect(within(detail).queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
    expect(api.installThemeListing).not.toHaveBeenCalled()
  })

  it('installs nothing until Add is clicked, then adds that listing by reference, once', async () => {
    api.installThemeListing.mockResolvedValue({ created: true, install: installRow() })
    renderLinked({ tab: 'themes', item: OWL_ID, add: '1' })

    const detail = await screen.findByTestId('theme-listing-detail')
    const add = await within(detail).findByRole('button', { name: ADD_TO_WORKSPACE })
    await within(detail).findByRole('region', { name: 'Versions' }) // the listing has loaded
    expect(api.installThemeListing).not.toHaveBeenCalled()

    fireEvent.click(add)

    await waitFor(() => expect(api.installThemeListing).toHaveBeenCalledWith(OWL_ID, undefined))
    expect(api.installThemeListing).toHaveBeenCalledTimes(1)
  })

  it('closing the dialog strips item and add from the address, the tab kept', async () => {
    renderLinked({ tab: 'themes', item: OWL_ID, add: '1' })

    const detail = await screen.findByTestId('theme-listing-detail')
    fireEvent.click(within(detail).getByRole('button', { name: 'Close' }))

    expect(nav.replace).toHaveBeenCalledWith('/marketplace?tab=themes')
    await waitFor(() => expect(screen.queryByTestId('theme-listing-detail')).not.toBeInTheDocument())
    expect(api.installThemeListing).not.toHaveBeenCalled()
  })
})

describe('what opens no dialog', () => {
  it('without add=1 the hero shows the listing and no dialog opens', async () => {
    renderLinked({ tab: 'themes', item: OWL_ID })

    await waitFor(async () => expect(await heroName()).toHaveTextContent('Night Owl'))
    expect(screen.queryByTestId('theme-listing-detail')).not.toBeInTheDocument()
    expect(nav.replace).not.toHaveBeenCalled()
  })

  it('an item that isn’t listed opens nothing, and the hero stays on the first listing', async () => {
    renderLinked({ tab: 'themes', item: 'nobody.here', add: '1' })

    expect(await heroName()).toHaveTextContent('Aurora')
    expect(screen.queryByTestId('theme-listing-detail')).not.toBeInTheDocument()
    expect(api.getThemeListing).not.toHaveBeenCalledWith('nobody.here')
    expect(api.installThemeListing).not.toHaveBeenCalled()
  })

  it('no link: the tab opens as before, on the first listing, with no dialog', async () => {
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    expect(await heroName()).toHaveTextContent('Aurora')
    expect(screen.queryByTestId('theme-listing-detail')).not.toBeInTheDocument()
    expect(nav.replace).not.toHaveBeenCalled()
  })
})
