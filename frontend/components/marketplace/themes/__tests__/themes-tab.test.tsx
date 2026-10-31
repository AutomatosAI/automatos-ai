/**
 * PRD-261 C-04b — the Marketplace's Themes tab (the storefront, 30 Oct 2026): the strip of listings,
 * the detail (the hero's Details), "Add to my workspace" then "Use this theme", Publish theme in its
 * dialog, and A-07d's Install theme and installed list reused on it. The hero itself, its looks,
 * its phone frame and the installed rows' View are storefront.test.tsx.
 *
 * `@clerk/nextjs` is A-07d's double (components/settings/themes/__tests__/install-update.test.tsx):
 * it does what Clerk's `createReverificationHandler` does without `onNeedsReverification`, so a
 * step-up 403 from the listing install is proved to reach Clerk from the component.
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
const clerk = vi.hoisted(() => ({ openReverification: vi.fn() }))
const workspace = vi.hoisted(() => ({ role: 'owner' as string }))
const style = vi.hoisted(() => ({ setStyle: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: workspace.role } }),
}))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => style }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', async () => {
  const { isReverificationHint } = await import('@clerk/shared/authorization-errors')
  type Fetcher = (...args: unknown[]) => Promise<unknown>
  return {
    useReverification: (fetcher: Fetcher) => async (...args: unknown[]) => {
      let result = await fetcher(...args)
      if (isReverificationHint(result)) {
        await clerk.openReverification()
        result = await fetcher(...args)
      }
      return result
    },
  }
})

import { INSTALL_THEME } from '@/components/settings/themes/install-theme'
import { USE_THEME } from '@/components/settings/themes/installed-row'
import {
  OPT_IN_WORDING,
  installRow,
  renderWithQuery,
  stepUpError,
  themeSettings,
} from '@/components/settings/themes/__tests__/theme-fixtures'

import {
  ADD_TO_WORKSPACE,
  INSTALLED_HEADING,
  MANAGERS_ADD,
  MARKETPLACE_OFF_NOTE,
  NOT_COMPATIBLE_HINT,
} from '../listing-format'
import { PENDING_REVIEW } from '../publish-result'
import { PUBLISH_THEME, SUBMIT_FOR_REVIEW } from '../publish-theme'
import { DETAILS, SUBMIT_PACKAGE } from '../storefront/copy'
import { MarketplaceThemesTab } from '../tab'
import { FILES, LISTED_PACKAGE_ID, LISTING_ID, LISTING_INSTALL_ID, listing, listingDetail } from './listing-fixtures'

/** The row A-07d's list shows once the listing install exists: reviewed, so no opt-in. */
const LISTED_ROW = installRow({
  installId: LISTING_INSTALL_ID, themeId: 'acme/aurora', name: 'Aurora', packageId: LISTED_PACKAGE_ID,
  version: '1.2.0', ownerType: 'marketplace', reviewed: true, private: false,
  optIn: { required: false, wording: OPT_IN_WORDING, packageId: LISTED_PACKAGE_ID },
})

const INSTALLED_STATE = {
  installId: LISTING_INSTALL_ID, packageId: LISTED_PACKAGE_ID, version: '1.2.0', fromListing: true, updatePackageId: null,
}

/** The hero opens on the first listing (Aurora); its Details opens C-04b's detail dialog. */
async function openDetails() {
  const hero = await screen.findByTestId('storefront-hero')
  fireEvent.click(within(hero).getByRole('button', { name: DETAILS }))
  return screen.findByTestId('theme-listing-detail')
}

async function openAurora() {
  renderWithQuery(<MarketplaceThemesTab searchQuery="" />)
  const detail = await openDetails()
  await within(detail).findByRole('region', { name: 'Versions' }) // the listing has loaded
  return detail
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
  clerk.openReverification.mockReset()
  clerk.openReverification.mockResolvedValue(undefined)
  style.setStyle.mockReset()
  workspace.role = 'owner'
  api.getThemeListings.mockResolvedValue({ listings: [listing()] })
  api.getThemeListing.mockResolvedValue(listingDetail())
  api.getInstalledThemes.mockResolvedValue({ installs: [] })
  api.getActiveTheme.mockResolvedValue({ choice: 'studio', theme: null })
  api.getThemeSettings.mockResolvedValue(themeSettings())
  api.putThemeChoice.mockResolvedValue({ choice: LISTING_INSTALL_ID })
})

describe('the strip of listings', () => {
  it('shows each approved theme with its id (its publisher’s) and version, and no count, rating or price', async () => {
    api.getThemeListings.mockResolvedValue({
      listings: [listing(), listing({ listingId: 'automatos.play', themeId: 'automatos/play', name: 'Play', publisher: { name: 'automatos', official: true } })],
    })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const card = await screen.findByTestId(`theme-listing-${LISTING_ID}`)
    expect(within(card).getByText('Aurora')).toBeInTheDocument()
    // The publisher is the theme id's first part (listings.py _publisher).
    expect(within(card).getByText('acme/aurora · 1.2.0')).toBeInTheDocument()
    expect(within(screen.getByTestId('theme-listing-automatos.play')).getByText('Official')).toBeInTheDocument()
    expect(within(card).queryByText('Free')).not.toBeInTheDocument()
    expect(within(card).queryByText(/install(s|ed) \d|rating|★/i)).not.toBeInTheDocument()
  })

  it('filters by the Marketplace search box', async () => {
    const owl = { listingId: 'nightco.owl', themeId: 'nightco/owl', name: 'Night Owl', description: null }
    api.getThemeListings.mockResolvedValue({ listings: [listing(), listing(owl)] })
    api.getThemeListing.mockResolvedValue(listingDetail(owl))
    renderWithQuery(<MarketplaceThemesTab searchQuery="owl" />)

    expect(await screen.findByTestId('theme-listing-nightco.owl')).toBeInTheDocument()
    expect(screen.queryByTestId(`theme-listing-${LISTING_ID}`)).not.toBeInTheDocument()
    expect(within(screen.getByTestId('storefront-hero')).getByTestId('storefront-hero-name')).toHaveTextContent('Night Owl')
  })

  it('shows the server’s sentence when the Marketplace refuses', async () => {
    const refusal = Object.assign(new Error('HTTP 403'), {
      status: 403, body: { detail: 'The theme Marketplace is turned off on this platform.' },
    })
    api.getThemeListings.mockRejectedValue(refusal)
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    expect(await screen.findByRole('alert')).toHaveTextContent('The theme Marketplace is turned off on this platform.')
  })

  it('never draws a cover from another origin', async () => {
    api.getThemeListings.mockResolvedValue({ listings: [listing({ cover: 'https://evil.example/cover.png' })] })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const card = await screen.findByTestId(`theme-listing-${LISTING_ID}`)
    expect(card.querySelector('img')).toBeNull()
  })
})

describe('the detail', () => {
  it('fetches the listing it opens and shows previews per preset, versions and what it changes', async () => {
    const detail = await openAurora()

    expect(api.getThemeListing).toHaveBeenCalledWith(LISTING_ID)
    const preset = await within(detail).findByTestId('theme-preset-default')
    const sources = Array.from(preset.querySelectorAll('img')).map((img) => img.getAttribute('src'))
    expect(sources).toEqual([`${FILES}/assets/previews/default.png`, `${FILES}/assets/previews/default-chat.png`])
    expect(within(detail).getByTestId('theme-version-1.2.0')).toHaveTextContent('Newest for this Automatos')
    expect(within(detail).getByTestId('theme-version-2.0.0-beta')).toHaveTextContent('Needs a newer Automatos')
    expect(within(detail).getByText('/chat')).toBeInTheDocument()
    expect(within(detail).getByRole('link', { name: 'Read me' })).toHaveAttribute('href', `${FILES}/README.md`)
  })

  it('a version this host can’t run can’t be added', async () => {
    api.getThemeListing.mockResolvedValue(listingDetail({ compatible: false, latest: null }))
    const detail = await openAurora()

    expect(await within(detail).findByRole('button', { name: ADD_TO_WORKSPACE })).toBeDisabled()
    expect(within(detail).getAllByText(NOT_COMPATIBLE_HINT).length).toBeGreaterThan(0)
  })
})

describe('Add to my workspace, then Use this theme', () => {
  it('installs the listing by reference, then chooses it without an opt-in (a reviewed version)', async () => {
    api.installThemeListing.mockResolvedValue({ created: true, install: LISTED_ROW })
    const detail = await openAurora()
    api.getInstalledThemes.mockResolvedValue({ installs: [LISTED_ROW] })
    api.getThemeListing.mockResolvedValue(listingDetail({ installed: INSTALLED_STATE }))

    fireEvent.click(await within(detail).findByRole('button', { name: ADD_TO_WORKSPACE }))
    await waitFor(() => expect(api.installThemeListing).toHaveBeenCalledWith(LISTING_ID, undefined))

    fireEvent.click(await within(detail).findByRole('button', { name: USE_THEME }))
    await waitFor(() => expect(api.putThemeChoice).toHaveBeenCalledWith(LISTING_INSTALL_ID))
    expect(style.setStyle).toHaveBeenCalledWith(`theme:install:${LISTING_INSTALL_ID}`)
    expect(screen.queryByTestId('theme-opt-in-dialog')).not.toBeInTheDocument()
    expect(api.optInTheme).not.toHaveBeenCalled()
  })

  it('a step-up 403 opens Clerk once and the install is sent again', async () => {
    api.installThemeListing.mockRejectedValueOnce(stepUpError()).mockResolvedValueOnce({ created: true, install: LISTED_ROW })
    const detail = await openAurora()

    fireEvent.click(await within(detail).findByRole('button', { name: ADD_TO_WORKSPACE }))
    await waitFor(() => expect(api.installThemeListing).toHaveBeenCalledTimes(2))
    expect(clerk.openReverification).toHaveBeenCalledTimes(1)
  })

  it('a workspace’s own upload of the theme opens A-07d’s opt-in dialog with the server’s wording', async () => {
    const ownUpload = installRow({ installId: LISTING_INSTALL_ID, themeId: 'acme/aurora', name: 'Aurora' })
    api.getInstalledThemes.mockResolvedValue({ installs: [ownUpload] })
    api.getThemeListing.mockResolvedValue(listingDetail({ installed: { ...INSTALLED_STATE, fromListing: false } }))
    const detail = await openAurora()

    fireEvent.click(await within(detail).findByRole('button', { name: USE_THEME }))
    const dialog = await screen.findByTestId('theme-opt-in-dialog')
    expect(within(dialog).getByTestId('theme-opt-in-wording')).toHaveTextContent(OPT_IN_WORDING)
    expect(api.putThemeChoice).not.toHaveBeenCalled()
  })
})

describe('who sees what', () => {
  it('an owner gets A-07d’s Install theme and installed list on the tab', async () => {
    api.getInstalledThemes.mockResolvedValue({ installs: [LISTED_ROW] })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const manage = await screen.findByRole('region', { name: INSTALLED_HEADING })
    expect(within(manage).getByRole('button', { name: INSTALL_THEME })).toBeInTheDocument()
    expect(within(manage).getByTestId('install-theme-input')).toHaveAttribute('accept', '.autotheme')
    const row = await within(manage).findByTestId(`installed-theme-${LISTING_INSTALL_ID}`)
    expect(within(row).getByTestId('theme-manage-actions')).toBeInTheDocument()
  })

  it('uninstalling a listing install in A-07d’s list refreshes the card (one query key tree)', async () => {
    api.getInstalledThemes.mockResolvedValue({ installs: [LISTED_ROW] })
    api.getThemeListings.mockResolvedValue({ listings: [listing({ installed: INSTALLED_STATE })] })
    api.uninstallTheme.mockResolvedValue({ uninstalled: true })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const card = await screen.findByTestId(`theme-listing-${LISTING_ID}`)
    expect(within(card).getByText('In this workspace')).toBeInTheDocument()
    api.getInstalledThemes.mockResolvedValue({ installs: [] })
    api.getThemeListings.mockResolvedValue({ listings: [listing()] })

    const row = await screen.findByTestId(`installed-theme-${LISTING_INSTALL_ID}`)
    fireEvent.click(within(row).getByRole('button', { name: 'Uninstall' }))
    fireEvent.click(within(row).getByRole('button', { name: 'Confirm uninstall' }))

    await waitFor(() => expect(api.uninstallTheme).toHaveBeenCalledWith(LISTING_INSTALL_ID))
    await waitFor(() => expect(within(screen.getByTestId(`theme-listing-${LISTING_ID}`)).queryByText('In this workspace')).not.toBeInTheDocument())
    expect(api.getThemeListings).toHaveBeenCalledTimes(2)
  })

  it('a member sees the listing and the list, but no Install theme, no Add and no manage actions', async () => {
    workspace.role = 'member'
    api.getInstalledThemes.mockResolvedValue({ installs: [LISTED_ROW] })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    // Checked before the detail opens: an open dialog hides the rest of the page from role queries.
    const row = await screen.findByTestId(`installed-theme-${LISTING_INSTALL_ID}`)
    expect(within(row).queryByTestId('theme-manage-actions')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: INSTALL_THEME })).not.toBeInTheDocument()
    expect(screen.queryByTestId('install-theme-input')).not.toBeInTheDocument()

    // The storefront waits for the theme settings (the marketplace switch), so it can come after the list.
    const hero = await screen.findByTestId('storefront-hero')
    expect(within(hero).getByText(MANAGERS_ADD)).toBeInTheDocument()
    expect(within(hero).queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
    const detail = await openDetails()
    expect(await within(detail).findByText(MANAGERS_ADD)).toBeInTheDocument()
    expect(within(detail).queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
    expect(api.installThemeListing).not.toHaveBeenCalled()
  })
})

describe('Publish theme on the tab', () => {
  const SUBMISSION = {
    package: {
      id: 'pkg-new', themeId: 'acme/hello', version: '1.0.0', name: 'Hello theme', digest: 'b'.repeat(64),
      ownerType: 'marketplace', status: 'pending', reviewed: false, scanVerdict: 'pass', scanReport: null,
    },
    duplicate: false,
  }

  it('every member gets C-01b’s form from Make your own, and it submits through apiClient.submitTheme', async () => {
    workspace.role = 'member'
    api.submitTheme.mockResolvedValue(SUBMISSION)
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    expect(screen.queryByRole('region', { name: PUBLISH_THEME })).not.toBeInTheDocument()
    fireEvent.click(await screen.findByRole('button', { name: SUBMIT_PACKAGE }))
    const dialog = await screen.findByTestId('publish-theme-dialog')
    const publish = within(dialog).getByRole('region', { name: PUBLISH_THEME })
    const file = new File(['zip'], 'hello.autotheme', { type: 'application/zip' })
    fireEvent.change(within(publish).getByTestId('publish-theme-input'), { target: { files: [file] } })
    fireEvent.click(within(publish).getByRole('button', { name: SUBMIT_FOR_REVIEW }))

    await waitFor(() => expect(api.submitTheme).toHaveBeenCalledWith(file))
    expect(await within(publish).findByText(PENDING_REVIEW)).toBeInTheDocument()
  })
})

describe('the Marketplace switch', () => {
  it('off: no storefront, no Publish theme and no listing request; private uploads and the list stay', async () => {
    api.getThemeSettings.mockResolvedValue(themeSettings({ marketplace: 'off' }))
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    expect(await screen.findByTestId('themes-marketplace-off')).toHaveTextContent(MARKETPLACE_OFF_NOTE)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: PUBLISH_THEME })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: SUBMIT_PACKAGE })).not.toBeInTheDocument()
    expect(screen.queryByTestId('theme-storefront')).not.toBeInTheDocument()
    expect(screen.queryByTestId('theme-listing-strip')).not.toBeInTheDocument()
    const manage = screen.getByRole('region', { name: INSTALLED_HEADING })
    expect(within(manage).getByRole('button', { name: INSTALL_THEME })).toBeInTheDocument()
    expect(api.getThemeListings).not.toHaveBeenCalled()
  })

  it('on: the storefront and the way to Publish theme show, and no off note', async () => {
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    expect(await screen.findByTestId(`theme-listing-${LISTING_ID}`)).toBeInTheDocument()
    expect(screen.getByTestId('storefront-hero')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: SUBMIT_PACKAGE }))
    expect(await screen.findByRole('region', { name: PUBLISH_THEME })).toBeInTheDocument()
    expect(screen.queryByTestId('themes-marketplace-off')).not.toBeInTheDocument()
  })
})

describe('who installs', () => {
  it('an admin left out by "who installs" gets the reason instead of Add to my workspace', async () => {
    workspace.role = 'admin'
    api.getThemeSettings.mockResolvedValue(themeSettings({ canInstall: false, installRole: 'owner' }))
    const detail = await openAurora()

    expect(await within(detail).findByTestId('theme-listing-install-refused')).toHaveTextContent(
      'In this workspace only the workspace owner can install themes.',
    )
    expect(within(detail).queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
    expect(api.installThemeListing).not.toHaveBeenCalled()
  })
})
