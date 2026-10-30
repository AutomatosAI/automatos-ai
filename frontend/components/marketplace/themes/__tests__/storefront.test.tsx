/**
 * PRD-261 — the Themes tab's storefront (the approved design, 30 Oct 2026): the hero for the selected
 * listing (its words, badges, looks and devices), Previous / Next, the dots and the strip's cards,
 * install in the hero (C-04b's AddToWorkspace, its gates unchanged), the empty stage, "Make your
 * own", and the installed rows' "Official x.y.z is listed" with View.
 *
 * The doubles are themes-tab.test.tsx's: apiClient, the workspace and its role, the style context
 * the chooser writes to, and Clerk's reverification (A-07d's double).
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

import { installRow, renderWithQuery, themeSettings } from '@/components/settings/themes/__tests__/theme-fixtures'
import type { ThemeListing, ThemeListingDetail } from '@/lib/api-client'

import { ADD_TO_WORKSPACE, MANAGERS_ADD, NOT_COMPATIBLE, NO_LISTINGS, NO_MATCHES, OFFICIAL_BADGE } from '../listing-format'
import { PUBLISH_THEME } from '../publish-theme'
import {
  DESIGN_KIT_URL,
  EMPTY_TITLE,
  LOOKS_HEADING,
  NEXT_THEME,
  OPEN_DESIGN_KIT,
  PREVIOUS_THEME,
  SUBMIT_PACKAGE,
  officialListed,
  showTheme,
  viewOfficial,
} from '../storefront/copy'
import { MarketplaceThemesTab } from '../tab'
import { FILES, LISTING_ID, listing, listingDetail } from './listing-fixtures'

const WEB = `${FILES}/assets/previews`
const PHONE = `${FILES}/assets/previews/phone`
const HUD_ID = 'automatos.hud'
const HUD_INSTALL_ID = '77777777-7777-4777-8777-777777777777'

const HUD: Partial<ThemeListing> = {
  listingId: HUD_ID, themeId: 'automatos/hud', name: 'HUD', version: '1.1.2',
  description: 'Automatos as a command deck.', publisher: { name: 'automatos', official: true },
  cover: `${WEB}/arc.png`, phoneCover: `${PHONE}/arc.png`,
}

const HUD_LOOKS: Partial<ThemeListingDetail> = {
  presets: [
    {
      id: 'arc', default: true,
      previews: [{ screen: null, url: `${WEB}/arc.png` }, { screen: 'chat', url: `${WEB}/arc-chat.png` }],
      phonePreviews: [{ screen: null, url: `${PHONE}/arc.png` }],
    },
    {
      id: 'forge', default: false,
      previews: [{ screen: null, url: `${WEB}/forge.png` }],
      phonePreviews: [{ screen: null, url: `${PHONE}/forge.png` }],
    },
    { id: 'ghost', default: false, previews: [{ screen: null, url: `${WEB}/ghost.png` }] },
  ],
}

/** Listings, and each one's detail by id, as C-04a answers them. */
function serve(summaries: ThemeListing[], details: Record<string, Partial<ThemeListingDetail>> = {}) {
  api.getThemeListings.mockResolvedValue({ listings: summaries })
  api.getThemeListing.mockImplementation(async (id: string) => {
    const summary = summaries.find((candidate) => candidate.listingId === id)
    return listingDetail({ ...summary, ...details[id] })
  })
}

const named = (id: string, name: string) => listing({ listingId: `${id}.theme`, themeId: `${id}/theme`, name })

async function hero() {
  return screen.findByTestId('storefront-hero')
}

async function heroName() {
  return within(await hero()).getByTestId('storefront-hero-name')
}

function webScreen(stage: HTMLElement) {
  return within(stage).getByTestId('storefront-web-screen')
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
  workspace.role = 'owner'
  serve([listing()])
  api.getInstalledThemes.mockResolvedValue({ installs: [] })
  api.getActiveTheme.mockResolvedValue({ choice: 'studio', theme: null })
  api.getThemeSettings.mockResolvedValue(themeSettings())
})

describe('the hero', () => {
  it('shows the first listing: its place, name, badges, id and version, a warning and its description', async () => {
    serve([
      listing({ ...HUD, compatible: false, revoked: { level: 'warn', reason: 'A fix is on its way.' } }),
      listing(),
    ])
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    expect(within(stage).getByTestId('storefront-hero-name')).toHaveTextContent('HUD')
    expect(within(stage).getByText('Featured · 1 of 2')).toBeInTheDocument()
    expect(within(stage).getByText(OFFICIAL_BADGE)).toBeInTheDocument()
    expect(within(stage).getByText(NOT_COMPATIBLE)).toBeInTheDocument()
    expect(within(stage).getByText('automatos/hud · 1.1.2')).toBeInTheDocument()
    expect(within(stage).getByText('A fix is on its way.')).toBeInTheDocument()
    expect(within(stage).getByText('Automatos as a command deck.')).toBeInTheDocument()
    expect(await within(stage).findByRole('button', { name: ADD_TO_WORKSPACE })).toBeDisabled()
  })

  it('switches the web and phone screens together with the looks chips, and a new listing starts on its first look', async () => {
    serve([listing(HUD), named('acme', 'Ink')], { [HUD_ID]: HUD_LOOKS })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    const looks = await within(stage).findByRole('group', { name: LOOKS_HEADING })
    expect(within(looks).getAllByRole('button').map((chip) => chip.textContent)).toEqual(['Arc', 'Arc · chat', 'Forge', 'Ghost'])
    expect(within(looks).getByRole('button', { name: 'Arc' })).toHaveAttribute('aria-pressed', 'true')
    expect(webScreen(stage)).toHaveAttribute('src', `${WEB}/arc.png`)

    fireEvent.click(within(looks).getByRole('button', { name: 'Forge' }))
    expect(within(looks).getByRole('button', { name: 'Forge' })).toHaveAttribute('aria-pressed', 'true')
    expect(within(looks).getByRole('button', { name: 'Arc' })).toHaveAttribute('aria-pressed', 'false')
    expect(webScreen(stage)).toHaveAttribute('src', `${WEB}/forge.png`)
    expect(within(within(stage).getByTestId('storefront-phone')).getByRole('img')).toHaveAttribute('src', `${PHONE}/forge.png`)

    fireEvent.click(within(stage).getByRole('button', { name: NEXT_THEME }))
    fireEvent.click(within(stage).getByRole('button', { name: PREVIOUS_THEME }))
    const again = await within(stage).findByRole('group', { name: LOOKS_HEADING })
    expect(within(again).getByRole('button', { name: 'Arc' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('moves with Previous, Next (wrapping), a dot and a card', async () => {
    serve([named('one', 'One'), named('two', 'Two'), named('three', 'Three')])
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    expect(await heroName()).toHaveTextContent('One')
    fireEvent.click(within(stage).getByRole('button', { name: NEXT_THEME }))
    expect(await heroName()).toHaveTextContent('Two')
    fireEvent.click(within(stage).getByRole('button', { name: showTheme('Three') }))
    expect(await heroName()).toHaveTextContent('Three')
    expect(within(stage).getByRole('button', { name: showTheme('Three') })).toHaveAttribute('aria-current', 'true')
    expect(within(stage).getByRole('button', { name: showTheme('One') })).not.toHaveAttribute('aria-current')
    fireEvent.click(within(stage).getByRole('button', { name: NEXT_THEME }))
    expect(await heroName()).toHaveTextContent('One')
    fireEvent.click(within(stage).getByRole('button', { name: PREVIOUS_THEME }))
    expect(await heroName()).toHaveTextContent('Three')

    const card = screen.getByTestId('theme-listing-two.theme')
    fireEvent.click(card)
    expect(await heroName()).toHaveTextContent('Two')
    expect(card).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('theme-listing-three.theme')).toHaveAttribute('aria-pressed', 'false')
  })

  it('has no Previous, Next or dots for one listing', async () => {
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    expect(within(stage).queryByRole('button', { name: NEXT_THEME })).not.toBeInTheDocument()
    expect(within(stage).queryByTestId('storefront-dots')).not.toBeInTheDocument()
  })

  it('loads its own screens at once, and the chips’ and the strip’s lazily', async () => {
    serve([listing(HUD)], { [HUD_ID]: HUD_LOOKS })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    const looks = await within(stage).findByRole('group', { name: LOOKS_HEADING })
    expect(webScreen(stage)).not.toHaveAttribute('loading', 'lazy')
    expect(within(within(stage).getByTestId('storefront-phone')).getByRole('img')).not.toHaveAttribute('loading', 'lazy')
    for (const img of looks.querySelectorAll('img')) expect(img).toHaveAttribute('loading', 'lazy')
    const card = screen.getByTestId(`theme-listing-${HUD_ID}`)
    expect(card.querySelectorAll('img')).toHaveLength(2) // the cover and the phone peek
    for (const img of card.querySelectorAll('img')) expect(img).toHaveAttribute('loading', 'lazy')
  })

  it('never draws a screen from another origin', async () => {
    const elsewhere = 'https://evil.example/shot.png'
    serve([listing({ cover: elsewhere, phoneCover: elsewhere })], {
      [LISTING_ID]: {
        description: 'The detail has answered.',
        presets: [{ id: 'arc', default: true, previews: [{ screen: null, url: elsewhere }], phonePreviews: [{ screen: null, url: elsewhere }] }],
        otherPreviews: [elsewhere],
        otherPhonePreviews: [elsewhere],
      },
    })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    expect(await within(stage).findByText('The detail has answered.')).toBeInTheDocument()
    expect(stage.querySelector('img')).toBeNull()
    expect(within(stage).queryByTestId('storefront-phone')).not.toBeInTheDocument()
    expect(screen.getByTestId(`theme-listing-${LISTING_ID}`).querySelector('img')).toBeNull()
  })
})

describe('the phone frame', () => {
  it('shows the look’s own phone screenshot, else the listing’s phone cover', async () => {
    serve([listing(HUD)], { [HUD_ID]: { ...HUD_LOOKS, phoneCover: `${PHONE}/cover.png` } })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    const looks = await within(stage).findByRole('group', { name: LOOKS_HEADING })
    const phone = () => within(within(stage).getByTestId('storefront-phone')).getByRole('img')
    expect(phone()).toHaveAttribute('src', `${PHONE}/arc.png`)
    fireEvent.click(within(looks).getByRole('button', { name: 'Ghost' }))
    expect(phone()).toHaveAttribute('src', `${PHONE}/cover.png`)
  })

  it('is not drawn for a listing without phone screenshots', async () => {
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    await within(stage).findByRole('group', { name: LOOKS_HEADING }) // the detail has answered
    expect(within(stage).queryByTestId('storefront-phone')).not.toBeInTheDocument()
    expect(webScreen(stage)).toHaveAttribute('src', `${FILES}/assets/previews/default.png`)
  })
})

describe('install in the hero', () => {
  it('an owner adds the listing from the hero', async () => {
    api.installThemeListing.mockResolvedValue({ created: true, install: installRow() })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    fireEvent.click(await within(stage).findByRole('button', { name: ADD_TO_WORKSPACE }))
    await waitFor(() => expect(api.installThemeListing).toHaveBeenCalledWith(LISTING_ID, undefined))
  })

  it('a member reads who adds themes, and gets no Add button', async () => {
    workspace.role = 'member'
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const stage = await hero()
    expect(within(stage).getByText(MANAGERS_ADD)).toBeInTheDocument()
    expect(within(stage).queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
  })
})

describe('nothing listed, or nothing found', () => {
  it('with nothing listed, the stage invites the first theme: the Design Kit and Submit a theme package', async () => {
    serve([])
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const empty = await screen.findByTestId('storefront-empty')
    expect(within(empty).getByText(NO_LISTINGS)).toBeInTheDocument()
    expect(within(empty).getByText(EMPTY_TITLE)).toBeInTheDocument()
    expect(within(empty).getByRole('link', { name: OPEN_DESIGN_KIT })).toHaveAttribute('href', DESIGN_KIT_URL)
    expect(screen.queryByTestId('storefront-hero')).not.toBeInTheDocument()
    expect(screen.queryByTestId('theme-listing-strip')).not.toBeInTheDocument()

    fireEvent.click(within(empty).getByRole('button', { name: SUBMIT_PACKAGE }))
    const dialog = await screen.findByTestId('publish-theme-dialog')
    expect(within(dialog).getByRole('region', { name: PUBLISH_THEME })).toBeInTheDocument()
  })

  it('when the search finds nothing, says so and keeps Make your own', async () => {
    renderWithQuery(<MarketplaceThemesTab searchQuery="nothing like it" />)

    expect(await screen.findByText(NO_MATCHES)).toBeInTheDocument()
    expect(screen.queryByTestId('storefront-hero')).not.toBeInTheDocument()
    expect(within(screen.getByTestId('theme-listing-strip')).getByTestId('storefront-make-your-own')).toBeInTheDocument()
  })
})

describe('Make your own', () => {
  it('ends the strip with the Design Kit and Submit a theme package, which opens Publish theme', async () => {
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const strip = await screen.findByTestId('theme-listing-strip')
    const card = within(strip).getByTestId('storefront-make-your-own')
    expect(strip.lastElementChild).toBe(card)
    const kit = within(card).getByRole('link', { name: OPEN_DESIGN_KIT })
    expect(kit).toHaveAttribute('href', DESIGN_KIT_URL)
    expect(kit).toHaveAttribute('target', '_blank')
    expect(kit).toHaveAttribute('rel', 'noopener noreferrer')
    expect(screen.queryByRole('region', { name: PUBLISH_THEME })).not.toBeInTheDocument()

    fireEvent.click(within(card).getByRole('button', { name: SUBMIT_PACKAGE }))
    const dialog = await screen.findByTestId('publish-theme-dialog')
    expect(within(dialog).getByRole('heading', { name: PUBLISH_THEME })).toBeInTheDocument()
    expect(within(dialog).getByTestId('publish-theme')).toBeInTheDocument()
  })
})

describe('installed in this workspace', () => {
  const privateHud = installRow({ installId: HUD_INSTALL_ID, themeId: 'automatos/hud', name: 'HUD', version: '1.0.0' })

  it('a private install of a theme with an Official listing says so, and View puts that listing in the hero', async () => {
    serve([listing(), listing(HUD)])
    api.getInstalledThemes.mockResolvedValue({ installs: [privateHud] })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    expect(await heroName()).toHaveTextContent('Aurora')
    const row = await screen.findByTestId(`installed-theme-${HUD_INSTALL_ID}`)
    expect(within(row).getByText(officialListed('1.1.2'))).toBeInTheDocument()
    expect(row.querySelector('img')).toHaveAttribute('src', `${WEB}/arc.png`)

    fireEvent.click(within(row).getByRole('button', { name: viewOfficial('HUD') }))
    expect(await heroName()).toHaveTextContent('HUD')
    expect(screen.getByTestId(`theme-listing-${HUD_ID}`)).toHaveAttribute('aria-pressed', 'true')
  })

  it('no notice for a listing install, or for a private install of a theme with no Official listing', async () => {
    const listed = installRow({ installId: HUD_INSTALL_ID, themeId: 'automatos/hud', name: 'HUD', private: false, reviewed: true, ownerType: 'marketplace' })
    const ownAurora = installRow({ themeId: 'acme/aurora', name: 'Aurora' })
    const unlisted = installRow({ installId: '88888888-8888-4888-8888-888888888888', themeId: 'acme/hello' })
    serve([listing(), listing(HUD)])
    api.getInstalledThemes.mockResolvedValue({ installs: [listed, ownAurora, unlisted] })
    renderWithQuery(<MarketplaceThemesTab searchQuery="" />)

    const hudRow = await screen.findByTestId(`installed-theme-${HUD_INSTALL_ID}`)
    await screen.findByTestId(`theme-listing-${HUD_ID}`) // the listings have answered
    expect(screen.queryByTestId('installed-official-listed')).not.toBeInTheDocument()
    expect(hudRow.querySelector('img')).toHaveAttribute('src', `${WEB}/arc.png`)
    expect(screen.getByTestId(`installed-theme-${unlisted.installId}`).querySelector('img')).toBeNull()
  })
})
