/**
 * PRD-260 T-008a — Make it yours at /looks, ported from Automatos Play's
 * tests/looks/LooksScreen.test.tsx onto the core's providers. Every control
 * goes through useLook(), so the look's variables on <html> (Play's and
 * Studio's) change at once and every page follows, the preview included.
 * Export writes the current look's JSON; Import goes through
 * lib/looks/transfer.ts and shows its messages (look-manage.test.tsx has the
 * rest of 'This look' and 'Import a look').
 * Under Classic or Studio the page is a short note with 'Switch to Play' (K9);
 * under Play below 1024 px, a note that it opens on a wider screen (D7).
 * Until the stored look is read (on the server, and in the first moments in
 * the browser) the page is only its heading, aria-busy, never Ember standing
 * in for the user's look (M72).
 * Switch to Play hands focus to the screen's heading as the note goes
 * (P260-FX-22, M21).
 * The width is a real window's (window-width.ts) read by Play's width source,
 * usePlayLayout (P260-FX-25, M69): on a phone the editor never mounts at all.
 * Roles, attributes and text only (CSS modules are proxies under css: false).
 */
import React from 'react'
import { renderToString } from 'react-dom/server'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children, fullBleed }: { children: React.ReactNode; fullBleed?: boolean }) => (
    <div data-testid="main-layout" data-full-bleed={fullBleed ? 'true' : 'false'}>
      {children}
    </div>
  ),
}))

import { exportStarted } from '@/components/looks/look-manager'
import { PHONE_WIDTH_PX, stubWindowWidth, type WindowWidth } from '@/hooks/__tests__/window-width'
import { LooksScreen } from '@/components/looks/looks-screen'
import { clearRoot, lookNamesLeft, myLook } from '@/lib/looks/__tests__/helpers'
import { ACCENTS, GROUNDS, LIGHTS, SHELLS, TYPE_STYLES } from '@/lib/looks/catalogue'
import { GLOW_SCALE, rgba, safeGlow } from '@/lib/looks/css'
import { SAVED_SUB } from '@/lib/looks/editor'
import { AURORA, DAYLIGHT, EMBER, OBSIDIAN, PRESETS, WORKSHOP } from '@/lib/looks/presets'
import { LOOK_SCOPE_KEY } from '@/lib/looks/storage'
import { LOOK_FILE_MAX_BYTES, LOOK_FILE_TYPE, TOO_BIG, lookJson } from '@/lib/looks/transfer'
import { PLAY_HTML_CLASS, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

import {
  BLOB_URL,
  type DownloadStub,
  KEYS,
  blobText,
  card,
  freshBrowser,
  group,
  importer,
  lookOnPage,
  looksRow,
  looksTree,
  manage,
  renderLooks,
  rootVar,
  status,
  stored,
  stubDownloads,
} from './looks-harness'

type User = ReturnType<typeof userEvent.setup>

const layout = () => screen.getByTestId('main-layout')
const fileInput = () => importer().getByLabelText<HTMLInputElement>('Or choose a .json file')

/** A phone-wide window, while a test sets one. */
let phone: WindowWidth | null = null

beforeEach(() => {
  freshBrowser()
  // jsdom plays no media; refused saves and imports are logged as warnings.
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  phone?.restore()
  phone = null
  vi.restoreAllMocks()
  freshBrowser()
  clearRoot()
})

describe('Make it yours under Play', () => {
  it("shows the heading with the active look's name, the preview with Auto, the Looks row and the editor", async () => {
    renderLooks()
    await lookOnPage(EMBER)
    expect(layout()).toHaveAttribute('data-full-bleed', 'true')
    expect(screen.getByRole('heading', { level: 1, name: 'Make it yours.' })).toBeInTheDocument()
    expect(status()).toHaveTextContent(/^Ember$/)
    expect(screen.getByRole('link', { name: 'See the style sheet' })).toHaveAttribute('href', '/looks/style')

    const preview = within(screen.getByRole('region', { name: 'Preview' }))
    expect(preview.getByText('Ta-da.')).toBeInTheDocument()
    expect(preview.getByText('Can you make me a shortlist under £700?')).toBeInTheDocument()
    expect(preview.getByText('✓ Searched the web · 5 results')).toBeInTheDocument()
    expect(preview.getByText('ASUS Vivobook S 14')).toBeInTheDocument()
    expect(preview.getByText('Laptop shortlist')).toBeInTheDocument()
    expect(preview.getByText('Open')).toBeInTheDocument()
    expect(preview.getByText('Message Auto')).toBeInTheDocument()
    expect(preview.getByTestId('auto-stage')).toHaveAttribute('aria-label', 'Ta-da.')

    expect(looksRow().getAllByRole('button').map((button) => button.textContent)).toEqual(
      PRESETS.map((look) => `Aa${look.name} ${look.sub}`),
    )
    expect(card('Ember')).toHaveAttribute('aria-pressed', 'true')
    expect(card('Aurora')).toHaveAttribute('aria-pressed', 'false')

    const editor = within(screen.getByRole('complementary', { name: 'Edit the look' }))
    for (const name of ['Accent · Molten', 'Background', 'Type', "Auto's shell", "Auto's light · ring"]) {
      expect(editor.getByRole('group', { name })).toBeInTheDocument()
    }
    expect(editor.getByRole('slider', { name: 'Glow' })).toHaveValue('24')
    expect(editor.getByRole('button', { name: 'Save as new look' })).toBeInTheDocument()
    expect(editor.getByRole('button', { name: 'Reset' })).toBeEnabled()
  })

  it('choosing a preset applies it through useLook: the look on <html> changes and the choice is kept', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(card('Aurora'))
    await lookOnPage(AURORA)
    expect(rootVar('--pl-accent')).toBe(ACCENTS.mint.hex)
    expect(card('Aurora')).toHaveAttribute('aria-pressed', 'true')
    expect(card('Ember')).toHaveAttribute('aria-pressed', 'false')
    expect(status()).toHaveTextContent(/^Aurora$/)
    expect(stored(KEYS.active)).toEqual(AURORA)
  })

  it("changing the accent updates the preview: the accent it draws with, its label, and '· edited'", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(group(/^Accent/).getByRole('button', { name: 'Rose' }))
    expect(status()).toHaveTextContent(/^Ember · edited$/)
    expect(screen.getByRole('group', { name: 'Accent · Rose' })).toBeInTheDocument()
    expect(group(/^Accent/).getByRole('button', { name: 'Rose' })).toHaveAttribute('aria-pressed', 'true')
    expect(group(/^Accent/).getByRole('button', { name: 'Molten' })).toHaveAttribute('aria-pressed', 'false')
    // The preview (and every page) reads the look's variables on <html>: Play's and Studio's.
    await lookOnPage({ ...EMBER, accent: 'rose' })
    expect(rootVar('--pl-accent')).toBe(ACCENTS.rose.hex)
    expect(stored(KEYS.active)).toEqual({ ...EMBER, accent: 'rose' })
  })

  it('re-themes the whole app from every control at once', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)

    await user.click(group('Background').getByRole('button', { name: 'Paper' }))
    await waitFor(() => expect(rootVar('--pl-ground-bg')).toBe(GROUNDS.paper.bg))

    await user.click(group('Type').getByRole('button', { name: 'Playful' }))
    await waitFor(() => expect(rootVar('--pl-display-family')).toBe(TYPE_STYLES.round.family))

    await user.click(group("Auto's shell").getByRole('button', { name: 'Cobalt' }))
    await waitFor(() => expect(rootVar('--pl-auto-shell-filter')).toBe(SHELLS.cobalt.filter))

    await user.click(group(/^Auto's light/).getByRole('button', { name: 'Ice' }))
    await waitFor(() => expect(rootVar('--pl-auto-light')).toBe(LIGHTS.ice.hex))

    // Paper × Molten keeps its text at AA only up to its safe glow (M50): the slider stops there.
    const safe = safeGlow('paper', 'molten')
    const slider = screen.getByRole('slider', { name: 'Glow' })
    expect(slider).toHaveAttribute('max', String(safe))
    fireEvent.change(slider, { target: { value: '0' } })
    await waitFor(() => expect(rootVar('--pl-glow')).toBe(rgba(ACCENTS.molten.hex, 0)))
    expect(slider).toHaveAttribute('aria-valuetext', 'Off')
    expect(screen.getByText('Off')).toBeInTheDocument()

    fireEvent.change(slider, { target: { value: '31' } })
    await waitFor(() => expect(rootVar('--pl-glow')).toBe(rgba(ACCENTS.molten.hex, safe / GLOW_SCALE)))
    expect(slider).toHaveAttribute('aria-valuetext', `${safe}%`)

    fireEvent.change(slider, { target: { value: String(safe - 2) } })
    await waitFor(() => expect(rootVar('--pl-glow')).toBe(rgba(ACCENTS.molten.hex, (safe - 2) / GLOW_SCALE)))
    expect(slider).toHaveAttribute('aria-valuetext', `${safe - 2}%`)

    const edited = { ...EMBER, ground: 'paper', font: 'round', shell: 'cobalt', light: 'ice', glow: safe - 2 } as const
    expect(stored(KEYS.active)).toEqual(edited)
    await lookOnPage(edited)
  })

  it("reaches the looks in Gerard's screenshots: Ember, Night, and Obsidian with a cobalt Auto", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(group('Background').getByRole('button', { name: 'Night' }))
    await lookOnPage({ ...EMBER, ground: 'indigo' })
    await user.click(card('Obsidian'))
    await user.click(group("Auto's shell").getByRole('button', { name: 'Cobalt' }))
    expect(status()).toHaveTextContent(/^Obsidian · edited$/)
    await lookOnPage({ ...OBSIDIAN, shell: 'cobalt' })
    expect(rootVar('--pl-auto-shell-filter')).toBe(SHELLS.cobalt.filter)
    expect(within(screen.getByRole('region', { name: 'Preview' })).getByTestId('auto-stage')).toBeInTheDocument()
  })

  it("Save as new look adds 'My look 1' to the row and selects it", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(group(/^Accent/).getByRole('button', { name: 'Gold' }))
    await user.click(screen.getByRole('button', { name: 'Save as new look' }))

    const saved = card('My look 1')
    expect(saved).toHaveAttribute('aria-pressed', 'true')
    expect(saved).toHaveTextContent(SAVED_SUB)
    expect(card('Ember')).toHaveAttribute('aria-pressed', 'false')
    expect(status()).toHaveTextContent(/^My look 1$/)

    const mine = { ...EMBER, id: 'look-1', name: 'My look 1', sub: SAVED_SUB, accent: 'gold' }
    expect(stored(KEYS.saved)).toEqual([mine])
    expect(stored(KEYS.active)).toEqual(mine)
    await waitFor(() => expect(rootVar('--pl-accent')).toBe(ACCENTS.gold.hex))

    await user.click(screen.getByRole('button', { name: 'Save as new look' }))
    expect(card('My look 2')).toHaveAttribute('aria-pressed', 'true')
    expect(looksRow().getAllByRole('button')).toHaveLength(PRESETS.length + 2)
  })

  it("Reset returns to the picked look's saved values", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(card('Workshop'))
    await user.click(group(/^Accent/).getByRole('button', { name: 'Mint' }))
    fireEvent.change(screen.getByRole('slider', { name: 'Glow' }), { target: { value: '40' } })
    expect(status()).toHaveTextContent(/^Workshop · edited$/)
    await waitFor(() => expect(rootVar('--pl-accent')).toBe(ACCENTS.mint.hex))

    await user.click(screen.getByRole('button', { name: 'Reset' }))
    expect(status()).toHaveTextContent(/^Workshop$/)
    await lookOnPage(WORKSHOP)
    expect(screen.getByRole('slider', { name: 'Glow' })).toHaveValue(String(WORKSHOP.glow))
    expect(stored(KEYS.active)).toEqual(WORKSHOP)
  })

  it('Reset on a saved look goes back to what was saved, not to a preset', async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(KEYS.saved, JSON.stringify([myLook()]))
    window.localStorage.setItem(KEYS.active, JSON.stringify(myLook({ accent: 'cream' })))
    renderLooks()
    await waitFor(() => expect(status()).toHaveTextContent(/^My look 1 · edited$/))
    await user.click(screen.getByRole('button', { name: 'Reset' }))
    expect(status()).toHaveTextContent(/^My look 1$/)
    await lookOnPage(myLook())
  })

  it("can't Reset a look that has nothing to go back to", async () => {
    window.localStorage.setItem(KEYS.active, JSON.stringify(myLook({ id: 'gone' })))
    renderLooks()
    await lookOnPage(myLook({ id: 'gone' }))
    expect(status()).toHaveTextContent(/^My look 1$/)
    expect(screen.getByRole('button', { name: 'Reset' })).toBeDisabled()
  })
})

describe('Make it yours before the stored look is read (M72)', () => {
  it('renders on the server as a busy frame with the heading, and no Ember card or option pressed', () => {
    window.localStorage.setItem(KEYS.active, JSON.stringify(DAYLIGHT))
    const html = renderToString(looksTree('play'))
    expect(html).toContain('aria-busy="true"')
    expect(html).toContain('Make it yours.')
    expect(html).not.toContain('aria-pressed="true"')
    expect(html).not.toContain(EMBER.sub)
    expect(html).not.toContain('Edit the look')
  })

  it('in the browser, shows the stored look once it is read, and is no longer busy', async () => {
    window.localStorage.setItem(KEYS.active, JSON.stringify(DAYLIGHT))
    renderLooks()
    await lookOnPage(DAYLIGHT)
    expect(card('Daylight')).toHaveAttribute('aria-pressed', 'true')
    expect(card('Ember')).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('heading', { level: 1, name: 'Make it yours.' }).closest('[aria-busy]')).toBeNull()
  })
})

describe('Export and Import through lib/looks/transfer.ts', () => {
  let stub: DownloadStub

  beforeEach(() => {
    stub = stubDownloads()
  })

  afterEach(() => stub.restore())

  it('Export produces the JSON of the current look, edits included, as <name>.look.json', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(group('Background').getByRole('button', { name: 'Paper' }))
    await user.click(manage().getByRole('button', { name: 'Export' }))

    expect(stub.downloads).toEqual([{ href: BLOB_URL, download: 'Ember.look.json' }])
    expect(stub.blobs[0]?.type).toBe(LOOK_FILE_TYPE)
    const text = await blobText(stub.blobs[0] as Blob)
    expect(text).toBe(lookJson({ ...EMBER, ground: 'paper' }))
    expect(JSON.parse(text)).toEqual({ ...EMBER, ground: 'paper' })
    // The Blob URL is let go DOWNLOAD_REVOKE_AFTER_MS later: wait for it while the stub is in place,
    // or the timer fires after afterEach has removed jsdom's missing revokeObjectURL.
    await waitFor(() => expect(stub.revoke).toHaveBeenCalledWith(BLOB_URL), { timeout: 2000 })
    expect(manage().getByText(exportStarted('Ember.look.json'))).toBeInTheDocument()
  })

  it("importing a bad file shows transfer's messages and adds nothing", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    const bad = JSON.stringify({ ...EMBER, schema: 'nope', accent: 'not-an-accent', script: 'alert(1)' })
    await user.upload(fileInput(), new File([bad], 'bad.look.json', { type: LOOK_FILE_TYPE }))

    const alert = await importer().findByRole('alert')
    expect(alert).toHaveTextContent("That look wasn't added.")
    expect(alert).toHaveTextContent(`"script" isn't part of a Look. Remove it.`)
    expect(alert).toHaveTextContent(`This isn't a v1 Look: "schema" must be "automatos-interface-look/v1".`)
    expect(alert).toHaveTextContent(`"accent" must be one of the catalogue's accents`)
    expect(looksRow().getAllByRole('button')).toHaveLength(PRESETS.length)
    expect(status()).toHaveTextContent(/^Ember$/)
    expect(stored(KEYS.saved)).toBeNull()
    await lookOnPage(EMBER)
  })

  it("importing a 1 MB file shows transfer's TOO_BIG before a byte is read", async () => {
    const user = userEvent.setup()
    const read = vi.spyOn(FileReader.prototype, 'readAsText')
    renderLooks()
    await lookOnPage(EMBER)
    const big = new File([new Uint8Array(LOOK_FILE_MAX_BYTES)], 'big.look.json', { type: LOOK_FILE_TYPE })
    await user.upload(fileInput(), big)

    expect(await importer().findByRole('alert')).toHaveTextContent(TOO_BIG)
    expect(read).not.toHaveBeenCalled()
    expect(stored(KEYS.saved)).toBeNull()
  })
})

describe('Make it yours without a LookProvider', () => {
  it("says in words why Save as new look didn't save, and logs it", async () => {
    const user = userEvent.setup()
    render(<LooksScreen />)
    await user.click(screen.getByRole('button', { name: 'Save as new look' }))
    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent("That look wasn't saved.")
    expect(alert).toHaveTextContent("Looks aren't available here.")
    expect(console.warn).toHaveBeenCalledWith('[looks] Save as new look was refused:', ["Looks aren't available here."])

    await user.click(looksRow().getByRole('button', { name: /^Daylight\b/ }))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

const OUTSIDE_PLAY: [string, UiStyle][] = [
  ['Studio', 'studio'],
  ['Classic', 'classic'],
]

describe('/looks outside Play (K9) and on a narrow screen (D7)', () => {
  it.each(OUTSIDE_PLAY)("under %s shows the note with 'Switch to Play', and no editor", (_label, style) => {
    renderLooks(style)
    expect(screen.getByRole('heading', { level: 1, name: 'Make it yours' })).toBeInTheDocument()
    // The note wears the core's frame: the shared PageHeader (PRD-244, app/__tests__/page-frames).
    expect(screen.getByTestId('page-title')).toHaveTextContent('Make it yours')
    expect(screen.getByText('Make it yours is part of the Play theme.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Switch to Play' })).toBeInTheDocument()
    expect(layout()).toHaveAttribute('data-full-bleed', 'false')
    expect(screen.queryByRole('complementary', { name: 'Edit the look' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Make it yours.' })).not.toBeInTheDocument()
    // Nothing of the look reaches <html> outside Play, and nothing is written, not even the pre-paint pointer.
    expect(lookNamesLeft()).toEqual([])
    expect(window.localStorage.getItem(KEYS.active)).toBeNull()
    expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBeNull()
  })

  it("'Switch to Play' is the menu's Play choice: the editor appears at once, on the default look", async () => {
    const user = userEvent.setup()
    renderLooks('studio')
    await user.click(screen.getByRole('button', { name: 'Switch to Play' }))
    expect(await screen.findByRole('heading', { level: 1, name: 'Make it yours.' })).toBeInTheDocument()
    expect(document.documentElement.classList.contains(PLAY_HTML_CLASS)).toBe(true)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=play`)
    expect(layout()).toHaveAttribute('data-full-bleed', 'true')
    expect(screen.queryByRole('button', { name: 'Switch to Play' })).not.toBeInTheDocument()
    await lookOnPage(EMBER)
  })

  it("'Switch to Play' hands focus to the screen's heading as the note goes, never to <body> (M21)", async () => {
    const user = userEvent.setup()
    renderLooks('studio')
    const switchToPlay = screen.getByRole('button', { name: 'Switch to Play' })
    switchToPlay.focus()
    await user.keyboard('{Enter}')
    const heading = await screen.findByRole('heading', { level: 1, name: 'Make it yours.' })
    expect(heading).toHaveFocus()
    expect(heading).toHaveAttribute('tabindex', '-1')
    expect(document.activeElement).not.toBe(document.body)
    await lookOnPage(EMBER)
  })

  it('under Play below 1024 px says it opens on a wider screen, with no editor and no switch', () => {
    phone = stubWindowWidth(PHONE_WIDTH_PX)
    renderLooks('play')
    expect(screen.getByTestId('page-title')).toHaveTextContent('Make it yours')
    expect(screen.getByText(/opens on a laptop or desktop screen/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Switch to Play' })).not.toBeInTheDocument()
    expect(screen.queryByRole('complementary', { name: 'Edit the look' })).not.toBeInTheDocument()
    expect(layout()).toHaveAttribute('data-full-bleed', 'false')
  })
})

/** Tab from the top until focus comes round again. */
async function tabStops(user: User): Promise<readonly Element[]> {
  const stops: Element[] = []
  for (let count = 0; count < 80; count += 1) {
    await user.tab()
    const focused = document.activeElement
    if (!focused || focused === document.body || stops.includes(focused)) break
    stops.push(focused)
  }
  return stops
}

describe('Make it yours by keyboard', () => {
  it('reaches every look, option, the slider and the actions with Tab', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    const controls = [...screen.getAllByRole('button'), screen.getByRole('slider', { name: 'Glow' })]
    const stops = await tabStops(user)
    expect(controls.filter((control) => !stops.includes(control)).map((control) => control.textContent)).toEqual([])
  })

  it('names every swatch and reads the slider value', async () => {
    renderLooks()
    await lookOnPage(EMBER)
    const accents = group(/^Accent/).getAllByRole('button')
    expect(accents.map((swatch) => swatch.getAttribute('aria-label'))).toEqual(
      Object.values(ACCENTS).map((accent) => accent.name),
    )
    expect(group('Background').getAllByRole('button').map((button) => button.textContent)).toEqual(
      Object.values(GROUNDS).map((ground) => ground.name),
    )
    expect(group('Type').getByRole('button', { name: 'Editorial' })).toHaveAttribute('aria-pressed', 'true')
    const slider = screen.getByRole('slider', { name: 'Glow' })
    expect(slider).toHaveAttribute('aria-valuetext', '24%')
    expect(slider).toHaveAttribute('min', '0')
    expect(slider).toHaveAttribute('max', String(safeGlow('graphite', 'molten')))
  })
})
