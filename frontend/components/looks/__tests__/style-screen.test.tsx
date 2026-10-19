/**
 * PRD-260 §4, K9 — the live style sheet at /looks/style, ported from Automatos
 * Play's tests/looks/StyleScreen.test.tsx. Under Play every value comes from
 * the active look and redraws when it changes (a test button changes it
 * through useLook, as Make it yours does); Auto's 12 states are the real
 * AutoMascot, each state's PNG until Play Auto's states plays their clips (never
 * under reduced motion), and none of them lights the page glow. The page goes through PlayOnlyRoute, as /looks does: under
 * Play at >= 1024 px the sheet is full bleed in MainLayout; under Classic or
 * Studio the page is the same note with Switch to Play (K9); under Play below
 * 1024 px, the same 'wider screen' note and no sheet (D7). MainLayout is a
 * pass-through here that records fullBleed (Play's frame has its own tests).
 * Until the stored look is read (on the server, and in the first moments in
 * the browser) the page is only its heading, aria-busy, and the Motion section
 * never says 'reduced motion' before the browser has answered (M72).
 * Switch to Play hands focus to the sheet's heading as the note goes
 * (P260-FX-22, M21).
 * The 12 specimens are decorative: no image named with a live status caption
 * ('Something went wrong'); each slot's visible name says the state
 * (P260-FX-24, M74). Copy tokens twice, or refused twice, is announced twice:
 * a new node each time (M22).
 * The width is a real window's (window-width.ts) read by Play's width source,
 * useThemeLayout (P260-FX-25, M69): on a phone the sheet never mounts at all.
 * Roles, attributes, inline custom properties and text only: CSS modules are
 * proxies.
 */
import React from 'react'
import { renderToString } from 'react-dom/server'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children, fullBleed }: { children: React.ReactNode; fullBleed?: boolean }) => (
    <div data-testid="main-layout" data-full-bleed={fullBleed ? 'true' : 'false'}>
      {children}
    </div>
  ),
}))
// PRD-261 A-04b: builtin Play's style sheet comes through its lazy slot (styleSheet@1); here as it
// is once its chunk has loaded, so the sheet renders in the same pass (and on the server).
vi.mock('@/components/looks/style/style-screen-lazy', async () => ({
  StyleScreen: (await import('@/components/looks/style/style-screen')).StyleScreen,
}))

import LookStylePage from '@/app/looks/style/page'
import { LookProvider, useLook } from '@/components/looks/look-provider'
import { SWITCH_TO_PLAY, WIDER_SCREEN_NOTE } from '@/components/looks/play-only-route'
import { SAMPLE_TITLE } from '@/components/looks/style/style-components'
import { PLAY_STATES } from '@/components/looks/style/style-mascots'
import { REDUCED_MOTION_NOTE, StyleMotion } from '@/components/looks/style/style-motion'
import { COPY_FAILED, COPY_TOKENS, STYLE_HEADING, STYLE_SHEET_NOTE } from '@/components/looks/style/style-screen'
import { BADGE_DONE } from '@/components/play/chat/receipts'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { LAPTOP_WIDTH_PX, PHONE_WIDTH_PX, stubWindowWidth } from '@/hooks/__tests__/window-width'
import { clearRoot, rootVars } from '@/lib/looks/__tests__/helpers'
import { ACCENTS, GROUNDS, LIGHTS } from '@/lib/looks/catalogue'
import { lookToCssVars } from '@/lib/looks/css'
import { lookDocumentVars } from '@/lib/looks/document'
import { AURORA, DAYLIGHT, EMBER } from '@/lib/looks/presets'
import { DEFAULT_LOOK_SCOPE } from '@/lib/looks/scope'
import { lookKeys } from '@/lib/looks/storage'
import { RADII, SPACING_STEPS, radiusToken, rootBlock } from '@/lib/looks/style'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import type { Look } from '@/lib/looks/types'
import { ringTone } from '@/lib/mascot/ring'
import { MASCOT_STATES } from '@/lib/mascot/types'
import { BUILTIN_PLAY_HTML_CLASS as PLAY_HTML_CLASS, BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  UI_STYLE_COOKIE,
  type UiStyle,
} from '@/lib/ui-style'

const ACCURACY = 0.05
const MASCOT_REGION = /^Mascot slot/
const REDUCED_MOTION = 'prefers-reduced-motion'

function Pick({ look }: { readonly look: Look }) {
  const { setLook } = useLook()
  return (
    <button type="button" onClick={() => setLook(look)}>
      {`Pick ${look.name}`}
    </button>
  )
}

function pageTree(style: UiStyle) {
  return (
    <UiStyleProvider initialStyle={style}>
      <LookProvider>
        <Pick look={DAYLIGHT} />
        <Pick look={AURORA} />
        <LookStylePage />
      </LookProvider>
    </UiStyleProvider>
  )
}

function renderPage(style: UiStyle = BUILTIN_PLAY_STYLE) {
  return render(pageTree(style))
}

/** The LookProvider has put the look on <html>, so storage is read and the sheet shows it. */
async function ready(look: Look = EMBER): Promise<void> {
  await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(look)))
}

function pick(look: Look): void {
  fireEvent.click(screen.getByRole('button', { name: `Pick ${look.name}` }))
}

function rowFor(label: string): HTMLElement {
  const row = screen.getByRole('rowheader', { name: label }).closest('tr')
  if (!row) throw new Error(`no row for ${label}`)
  return row
}

/** The ratio shown in the contrast table's row for a pair, e.g. "text / bg". */
function ratioFor(label: string): number {
  return Number.parseFloat(within(rowFor(label)).getByText(/^\d+\.\d{2}:1$/).textContent ?? '')
}

function verdictFor(label: string): string {
  return within(rowFor(label)).getByText(/AA (pass|fail)/).textContent ?? ''
}

/** The swatch for a look variable, e.g. "--pl-ground-bg": its chip, name, value, role and note. */
function swatchItem(name: string): HTMLElement {
  const item = screen.getByText(name).closest('li')
  if (!item) throw new Error(`no swatch ${name}`)
  return item
}

function swatchValue(name: string): string {
  return swatchItem(name).children[2]?.textContent ?? ''
}

/** The colour a swatch's chip is drawn in: the --pl-look-swatch swatch-style.ts's colourStyle sets. */
function chipColour(name: string): string {
  const chip = swatchItem(name).children[0]
  if (!(chip instanceof HTMLElement)) throw new Error(`no chip for ${name}`)
  return chip.style.getPropertyValue('--pl-look-swatch')
}

/** The inline custom property each item's drawn specimen carries, in a scale section. */
function specimenValues(section: string, property: string): string[] {
  const items = within(screen.getByRole('region', { name: section })).getAllByRole('listitem')
  return items.map((item) => {
    const specimen = item.firstElementChild
    return specimen instanceof HTMLElement ? specimen.style.getPropertyValue(property) : ''
  })
}

const layout = () => screen.getByTestId('main-layout')

function mascotRegion(): HTMLElement {
  return screen.getByRole('region', { name: MASCOT_REGION })
}

const playStates = () => within(mascotRegion()).getByRole('button', { name: PLAY_STATES })
const stillsIn = (region: HTMLElement) => Array.from(region.querySelectorAll('img'))
const clipsIn = (region: HTMLElement) => Array.from(region.querySelectorAll('video'))

const originalMatchMedia = Object.getOwnPropertyDescriptor(window, 'matchMedia')
const originalClipboard = Object.getOwnPropertyDescriptor(window.navigator, 'clipboard')

function preferReducedMotion(): void {
  const matchMedia = (media: string) => ({
    matches: media.includes(REDUCED_MOTION),
    media,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
  })
  Object.defineProperty(window, 'matchMedia', { configurable: true, writable: true, value: matchMedia })
}

function stubClipboard(writeText: (text: string) => Promise<void>): void {
  Object.defineProperty(window.navigator, 'clipboard', { configurable: true, value: { writeText } })
}

function restore(target: object, name: string, original: PropertyDescriptor | undefined): void {
  if (original) Object.defineProperty(target, name, original)
  else Reflect.deleteProperty(target, name)
}

beforeEach(() => {
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  clearRoot()
})

afterEach(() => {
  cleanup()
  restore(window, 'matchMedia', originalMatchMedia)
  restore(window.navigator, 'clipboard', originalClipboard)
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  clearRoot()
  vi.restoreAllMocks()
})

describe('the style sheet under Play, on Ember', () => {
  it('renders full bleed inside MainLayout with the live contrast ratios and an AA verdict in words', async () => {
    renderPage()
    await ready()
    expect(within(layout()).getByRole('heading', { level: 1, name: STYLE_HEADING })).toBeInTheDocument()
    expect(layout()).toHaveAttribute('data-full-bleed', 'true')
    expect(screen.queryByText(STYLE_SHEET_NOTE)).toBeNull()
    expect(Math.abs(ratioFor('text / bg') - 16.36)).toBeLessThanOrEqual(ACCURACY)
    expect(Math.abs(ratioFor('text-2 / bg') - 8.54)).toBeLessThanOrEqual(ACCURACY)
    expect(Math.abs(ratioFor('ink / molten') - 6.74)).toBeLessThanOrEqual(ACCURACY)
    expect(ratioFor('text / surface')).toBeGreaterThan(4.5)
    expect(verdictFor('text / bg')).toBe('✓ AA pass')
    expect(verdictFor('accent text / bg')).toBe('✓ AA pass')
  })

  it('shows a swatch for every colour with its name, value and role', async () => {
    renderPage()
    await ready()
    const colour = within(screen.getByRole('region', { name: 'Colour' }))
    expect(colour.getAllByRole('listitem')).toHaveLength(22)
    expect(swatchValue('--pl-ground-bg')).toBe(GROUNDS.graphite.bg)
    expect(chipColour('--pl-ground-bg')).toBe(GROUNDS.graphite.bg)
    expect(chipColour('--pl-accent')).toBe(ACCENTS.molten.hex)
    expect(within(swatchItem('--pl-ground-bg')).getByText('page')).toBeInTheDocument()
    expect(swatchValue('--pl-accent')).toBe(ACCENTS.molten.hex)
    expect(swatchValue('--pl-accent-text')).toBe(ACCENTS.molten.hex)
    expect(within(swatchItem('--pl-accent-text')).queryByText(/for reading on/)).toBeNull()
  })

  it('shows the type, spacing, radii and a components gallery that is only a picture', async () => {
    renderPage()
    await ready()
    const type = within(screen.getByRole('region', { name: 'Type' }))
    for (const text of ['Hero 72px', 'The fun way.', 'Searching.', 'Laptop shortlist', 'Mono 13']) {
      expect(type.getByText(text)).toBeInTheDocument()
    }
    expect(type.getByText(/Editorial · Instrument Serif italic 400/)).toBeInTheDocument()
    expect(within(screen.getByRole('region', { name: 'Spacing · 4px base' })).getAllByRole('listitem')).toHaveLength(7)
    expect(within(screen.getByRole('region', { name: 'Radii' })).getByText('pill chip')).toBeInTheDocument()
    const steps = SPACING_STEPS.map((step) => `var(${step.token})`)
    const radii = RADII.map((radius) => `var(${radiusToken(radius.kind)})`)
    expect(specimenValues('Spacing · 4px base', '--pl-specimen-step')).toEqual(steps)
    expect(specimenValues('Radii', '--pl-specimen-radius')).toEqual(radii)
    const components = within(screen.getByRole('region', { name: 'Components' }))
    for (const text of ['Send', 'Stop', 'All', 'Reading 4 pages', "Web search didn't answer", 'What Auto did', 'Open']) {
      expect(components.getByText(text)).toBeInTheDocument()
    }
    // "What Auto did": Play's Done tag beside the write, none beside the read (P260-FX-14).
    expect(components.getByText(`Made a report: ${SAMPLE_TITLE}`).nextElementSibling).toHaveTextContent(BADGE_DONE)
    expect(components.getByText('Searched the web · 5 results').nextElementSibling).toBeNull()
    expect(components.queryAllByRole('button')).toEqual([])
    expect(components.queryAllByRole('link')).toEqual([])
  })

  it("shows Auto's 12 states as AutoMascot, each ring in the look, none lighting the page glow", async () => {
    renderPage()
    await ready()
    const mascot = within(mascotRegion())
    expect(mascot.getAllByRole('listitem')).toHaveLength(MASCOT_STATES.length)
    const stages = mascot.getAllByTestId('auto-stage')
    expect(stages.map((stage) => stage.getAttribute('data-ring'))).toEqual(MASCOT_STATES.map((state) => ringTone(state)))
    expect(stages.filter((stage) => stage.hasAttribute('data-glow'))).toEqual([])
    expect(stillsIn(mascotRegion()).map((still) => still.getAttribute('src'))).toEqual(MASCOT_STATES.map((state) => `/auto/${state}.png`))
    expect(mascot.getAllByText(`ring light · ${LIGHTS.mint.hex}`)).toHaveLength(4)
    expect(mascot.getAllByText(`ring accent · ${ACCENTS.molten.hex}`)).toHaveLength(4)
  })

  it('names no specimen with a live status caption: the 12 slots name the states, Auto is decorative (P260-FX-24, M74)', async () => {
    renderPage()
    await ready()
    const region = mascotRegion()
    expect(screen.queryByRole('img', { name: 'Something went wrong' })).toBeNull()
    expect(screen.queryByRole('img', { name: "Automatos isn't running" })).toBeNull()
    expect(within(region).queryAllByRole('img')).toEqual([])
    for (const state of MASCOT_STATES) expect(within(region).getByText(state)).toBeInTheDocument()
    const stages = within(region).getAllByTestId('auto-stage')
    expect(stages).toHaveLength(MASCOT_STATES.length)
    expect(stages.filter((stage) => stage.hasAttribute('role') || stage.hasAttribute('aria-label'))).toEqual([])
    expect(stages.filter((stage) => stage.closest('[aria-hidden="true"]') === null)).toEqual([])
  })

  it("shows the 12 states as lazy stills, and Play Auto's states plays their 12 clips (P260-FX-15)", async () => {
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    renderPage()
    await ready()
    const region = mascotRegion()
    expect(within(region).getAllByTestId('auto-stage')).toHaveLength(MASCOT_STATES.length)
    expect(clipsIn(region)).toEqual([])
    expect(stillsIn(region).map((still) => still.getAttribute('src'))).toEqual(MASCOT_STATES.map((state) => `/auto/${state}.png`))
    expect(stillsIn(region).map((still) => still.getAttribute('loading'))).toEqual(MASCOT_STATES.map(() => 'lazy'))
    expect(playStates()).toHaveAttribute('aria-pressed', 'false')
    expect(playStates()).toBeEnabled()

    fireEvent.click(playStates())
    expect(playStates()).toHaveAttribute('aria-pressed', 'true')
    expect(clipsIn(region).map((clip) => clip.getAttribute('src'))).toEqual(MASCOT_STATES.map((state) => `/auto/${state}.webm`))
    expect(clipsIn(region).map((clip) => clip.getAttribute('poster'))).toEqual(MASCOT_STATES.map((state) => `/auto/${state}.png`))
    expect(stillsIn(region)).toEqual([])

    fireEvent.click(playStates())
    expect(playStates()).toHaveAttribute('aria-pressed', 'false')
    expect(clipsIn(region)).toEqual([])
  })

  it("shows every state's PNG under reduced motion, no clip, and Play Auto's states is off", async () => {
    preferReducedMotion()
    renderPage()
    await ready()
    const stills = stillsIn(mascotRegion())
    expect(stills.map((still) => still.getAttribute('src'))).toEqual(MASCOT_STATES.map((state) => `/auto/${state}.png`))
    expect(clipsIn(mascotRegion())).toHaveLength(0)
    expect(playStates()).toBeDisabled()
    expect(playStates()).toHaveAttribute('aria-pressed', 'false')
  })

  it('links back to Make it yours', async () => {
    renderPage()
    await ready()
    expect(screen.getByRole('link', { name: 'Back to Looks' })).toHaveAttribute('href', '/looks')
  })
})

describe('the style sheet follows the active look', () => {
  it('redraws the bg swatch and shows a darkened accent text on Daylight', async () => {
    renderPage()
    await ready()
    pick(DAYLIGHT)
    await ready(DAYLIGHT)
    expect(swatchValue('--pl-ground-bg')).toBe(GROUNDS.paper.bg)
    expect(chipColour('--pl-ground-bg')).toBe(GROUNDS.paper.bg)
    expect(swatchValue('--pl-accent-text')).not.toBe(ACCENTS.molten.hex)
    expect(swatchValue('--pl-accent-text')).toBe(lookToCssVars(DAYLIGHT)['--pl-accent-text'])
    expect(within(swatchItem('--pl-accent-text')).getByText('darkened for reading on Paper')).toBeInTheDocument()
    expect(within(swatchItem('--pl-error-text')).getByText('darkened for reading on Paper')).toBeInTheDocument()
    expect(ratioFor('accent text / bg')).toBeGreaterThanOrEqual(4.5)
  })

  it('redraws the type and the rings for another look', async () => {
    renderPage()
    await ready()
    pick(AURORA)
    await ready(AURORA)
    expect(screen.getByText('Hero 58px')).toBeInTheDocument()
    expect(screen.getByText(/Characterful · Bricolage Grotesque/)).toBeInTheDocument()
    expect(screen.getAllByText(`ring light · ${LIGHTS.ice.hex}`)).toHaveLength(4)
    expect(screen.getAllByText(`ring accent · ${ACCENTS.mint.hex}`)).toHaveLength(4)
    expect(screen.getByRole('rowheader', { name: 'ink / mint' })).toBeInTheDocument()
    expect(swatchValue('--pl-accent')).toBe(ACCENTS.mint.hex)
  })
})

describe('Copy tokens', () => {
  it("copies the active look's variables as a :root block, and the notice goes with the look", async () => {
    const writeText = vi.fn(async (_text: string) => undefined)
    stubClipboard(writeText)
    renderPage()
    await ready()
    fireEvent.click(screen.getByRole('button', { name: 'Copy tokens' }))
    expect(await screen.findByText("Copied Ember's tokens as a :root block.")).toBeInTheDocument()
    expect(writeText).toHaveBeenCalledWith(rootBlock(EMBER))
    pick(DAYLIGHT)
    await ready(DAYLIGHT)
    expect(screen.queryByText("Copied Ember's tokens as a :root block.")).toBeNull()
  })

  it('says it again when the same copy is made twice: a new node in the polite region (P260-FX-24, M22)', async () => {
    const writeText = vi.fn(async (_text: string) => undefined)
    stubClipboard(writeText)
    renderPage()
    await ready()
    const copied = "Copied Ember's tokens as a :root block."
    fireEvent.click(screen.getByRole('button', { name: COPY_TOKENS }))
    const first = await screen.findByText(copied)

    fireEvent.click(screen.getByRole('button', { name: COPY_TOKENS }))
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(first.isConnected).toBe(false))
    expect(screen.getByText(copied).closest('[aria-live="polite"]')).not.toBeNull()
  })

  it('puts the alert in again when the browser refuses the clipboard twice (P260-FX-24, M22)', async () => {
    stubClipboard(async () => {
      throw new DOMException('Write permission denied.', 'NotAllowedError')
    })
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    renderPage()
    await ready()
    fireEvent.click(screen.getByRole('button', { name: COPY_TOKENS }))
    const first = await screen.findByRole('alert')

    fireEvent.click(screen.getByRole('button', { name: COPY_TOKENS }))
    await waitFor(() => expect(first.isConnected).toBe(false))
    expect(screen.getByRole('alert')).toHaveTextContent(COPY_FAILED)
  })

  it('says so when the browser blocks the clipboard', async () => {
    const denied = new DOMException('Write permission denied.', 'NotAllowedError')
    stubClipboard(async () => {
      throw denied
    })
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    renderPage()
    await ready()
    fireEvent.click(screen.getByRole('button', { name: 'Copy tokens' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(COPY_FAILED)
    expect(console.error).toHaveBeenCalledWith('[looks] copying the tokens failed', denied)
  })
})

describe('Motion', () => {
  it('plays the notes with a toggle', async () => {
    renderPage()
    await ready()
    const motion = within(screen.getByRole('region', { name: 'Motion' }))
    expect(motion.getByText(/rise 12px/)).toBeInTheDocument()
    const toggle = motion.getByRole('button', { name: 'Play the motion' })
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
  })

  it('is off under reduced motion, and says why', async () => {
    preferReducedMotion()
    renderPage()
    await ready()
    const motion = within(screen.getByRole('region', { name: 'Motion' }))
    expect(motion.getByRole('button', { name: 'Play the motion' })).toBeDisabled()
    expect(motion.getByText(/reduced motion, so nothing plays/)).toBeInTheDocument()
  })
})

describe('/looks/style under Classic or Studio (K9), the same note as /looks', () => {
  it.each<UiStyle>(['studio', 'classic'])("shows PlayOnlyRoute's note and Switch to Play under %s, never the sheet", (style) => {
    renderPage(style)
    const note = within(screen.getByRole('region', { name: STYLE_HEADING }))
    expect(note.getByRole('heading', { level: 1, name: STYLE_HEADING })).toBeInTheDocument()
    expect(note.getByText(STYLE_SHEET_NOTE)).toBeInTheDocument()
    expect(note.getByRole('button', { name: SWITCH_TO_PLAY })).toBeInTheDocument()
    expect(layout()).toHaveAttribute('data-full-bleed', 'false')
    expect(screen.queryByRole('region', { name: 'Colour' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Copy tokens' })).toBeNull()
    expect(document.documentElement.style.getPropertyValue('--pl-ground-bg')).toBe('')
  })

  it('Switch to Play makes Play the style and the sheet renders in place, full bleed, on the look', async () => {
    renderPage('studio')
    fireEvent.click(screen.getByRole('button', { name: SWITCH_TO_PLAY }))
    expect(await screen.findByRole('region', { name: 'Colour' })).toBeInTheDocument()
    expect(screen.queryByText(STYLE_SHEET_NOTE)).toBeNull()
    expect(layout()).toHaveAttribute('data-full-bleed', 'true')
    expect(document.documentElement.classList.contains(PLAY_HTML_CLASS)).toBe(true)
    await ready()
  })
})

describe('/looks/style after Switch to Play, by keyboard (P260-FX-22, M21)', () => {
  it("hands focus to the sheet's heading as the note goes, never to <body>", async () => {
    renderPage('studio')
    const switchToPlay = screen.getByRole('button', { name: SWITCH_TO_PLAY })
    switchToPlay.focus()
    fireEvent.click(switchToPlay)
    expect(await screen.findByRole('region', { name: 'Colour' })).toBeInTheDocument()
    const heading = screen.getByRole('heading', { level: 1, name: STYLE_HEADING })
    expect(heading).toHaveFocus()
    expect(heading).toHaveAttribute('tabindex', '-1')
    expect(document.activeElement).not.toBe(document.body)
    await ready()
  })
})

describe('/looks/style under Play below 1024 px (D7), as /looks', () => {
  it('says it opens on a wider screen, with no sheet and no switch', () => {
    stubWindowWidth(PHONE_WIDTH_PX)
    renderPage(BUILTIN_PLAY_STYLE)
    const note = within(screen.getByRole('region', { name: STYLE_HEADING }))
    expect(note.getByText(WIDER_SCREEN_NOTE)).toBeInTheDocument()
    expect(screen.queryByText(STYLE_SHEET_NOTE)).toBeNull()
    expect(screen.queryByRole('button', { name: SWITCH_TO_PLAY })).toBeNull()
    expect(screen.queryByRole('region', { name: 'Colour' })).toBeNull()
    expect(screen.queryByRole('region', { name: MASCOT_REGION })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Copy tokens' })).toBeNull()
    expect(layout()).toHaveAttribute('data-full-bleed', 'false')
  })

  it('shows the sheet once the window is wide enough', async () => {
    const window_ = stubWindowWidth(PHONE_WIDTH_PX)
    renderPage(BUILTIN_PLAY_STYLE)
    expect(screen.queryByRole('region', { name: 'Colour' })).toBeNull()
    act(() => window_.resize(LAPTOP_WIDTH_PX))
    expect(await screen.findByRole('region', { name: 'Colour' })).toBeInTheDocument()
    expect(screen.queryByText(WIDER_SCREEN_NOTE)).toBeNull()
    expect(layout()).toHaveAttribute('data-full-bleed', 'true')
    await ready()
  })
})

describe('/looks/style before the stored look is read (M72)', () => {
  it('renders on the server as a busy frame with the heading: no sheet, no look named, no reduced-motion note', () => {
    window.localStorage.setItem(lookKeys(DEFAULT_LOOK_SCOPE).active, JSON.stringify(DAYLIGHT))
    const html = renderToString(pageTree(BUILTIN_PLAY_STYLE))
    expect(html).toContain('aria-busy="true"')
    expect(html).toContain(STYLE_HEADING)
    expect(html).not.toContain(COPY_TOKENS)
    expect(html).not.toContain('Every value here comes from')
    expect(html).not.toContain(REDUCED_MOTION_NOTE)
  })

  it("the Motion section waits for the browser's own answer before it says reduced motion", () => {
    // The server reads every browser as asking for reduced motion (hooks/use-media-preferences.ts).
    const html = renderToString(
      <LookProvider>
        <StyleMotion />
      </LookProvider>,
    )
    expect(html).toContain('Play the motion')
    expect(html).not.toContain(REDUCED_MOTION_NOTE)
  })

  it('in the browser, drops aria-busy once the stored look is read', async () => {
    window.localStorage.setItem(lookKeys(DEFAULT_LOOK_SCOPE).active, JSON.stringify(DAYLIGHT))
    renderPage()
    await ready(DAYLIGHT)
    const heading = screen.getByRole('heading', { level: 1, name: STYLE_HEADING })
    expect(heading.closest('[aria-busy]')).toBeNull()
    expect(screen.getByText(/Every value here comes from Daylight/)).toBeInTheDocument()
  })
})
