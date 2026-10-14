import { ringColour, ringTone, type RingTone } from '@/lib/mascot/ring'
import { MASCOT_STATES, type MascotName } from '@/lib/mascot/types'

import { ACCENTS, ERROR_COLOUR, GROUNDS, TYPE_STYLES } from './catalogue'
import { AA_TEXT, contrastRatio, relativeLuminance } from './contrast'
import { lookToCssVars, type CssVars } from './css'
import type { Look } from './types'

/**
 * PRD-260 §4 — the live style sheet (/looks/style) as data, ported from
 * Automatos Play's lib/looks/style.ts: the active look's colours with live
 * contrast, its type, the spacing and radii, the motion notes and Auto's 12
 * states. Pure: every value is computed from the catalogue and the look it is
 * given, so the sheet redraws whenever the active look changes. The names are
 * the core's --pl-* variables (K2): lib/looks/css.ts sets them, and
 * app/play-theme.css holds the tokens the type and scale are read from.
 */

export interface ColourToken {
  /** A look variable from lib/looks/css.ts. */
  readonly name: string
  readonly role: string
}

/** Every colour a look sets, with what it is for. lib/looks/__tests__/style.test.ts keeps it complete. */
export const COLOUR_TOKENS: readonly ColourToken[] = [
  { name: '--pl-ground-bg', role: 'page' },
  { name: '--pl-ground-surface', role: 'inputs, cards' },
  { name: '--pl-ground-text', role: 'text, the secondary button' },
  { name: '--pl-ground-text-2', role: 'secondary text' },
  { name: '--pl-ground-line', role: 'lines' },
  { name: '--pl-ground-chip', role: 'chips, finished tool lines' },
  { name: '--pl-ground-panel', role: 'panels' },
  { name: '--pl-accent', role: 'the one primary per screen, rings, icons' },
  { name: '--pl-accent-ink', role: 'text on the accent' },
  { name: '--pl-accent-text', role: 'the accent as text' },
  { name: '--pl-accent-line', role: 'selected and new edges' },
  { name: '--pl-accent-fill', role: 'a running tool line' },
  { name: '--pl-glow', role: 'the page glow' },
  { name: '--pl-error', role: 'errors: fills, rings, icons' },
  { name: '--pl-error-text', role: 'the error as text' },
  { name: '--pl-auto-light', role: "Auto's eyes, ring and connected only" },
  { name: '--pl-auto-shell', role: "Auto's shell" },
  { name: '--pl-auto-dusk', role: "Auto's ring asleep or lost" },
  { name: '--pl-sheet-bg', role: "a Deliverable's page" },
  { name: '--pl-sheet-text', role: 'text on that page' },
  { name: '--pl-sheet-accent', role: 'the accent on that page' },
]

export interface ColourSwatch extends ColourToken {
  readonly value: string
  /** Set when a text variant differs from its raw colour: "darkened for reading on Paper". */
  readonly note: string | null
}

function valueOf(vars: CssVars, name: string): string {
  const value = vars[name]
  if (value === undefined) throw new Error(`A look sets no ${name}`)
  return value
}

/** Why a readable variant differs from its raw colour, or null when it is the raw colour. */
export function variantNote(raw: string, derived: string, groundName: string): string | null {
  if (raw.toUpperCase() === derived.toUpperCase()) return null
  const way = relativeLuminance(derived) < relativeLuminance(raw) ? 'darkened' : 'lightened'
  return `${way} for reading on ${groundName}`
}

export function colourSwatches(look: Look): readonly ColourSwatch[] {
  const vars = lookToCssVars(look)
  const groundName = GROUNDS[look.ground].name
  const raws: Readonly<Record<string, string>> = {
    '--pl-accent-text': ACCENTS[look.accent].hex,
    '--pl-error-text': ERROR_COLOUR,
  }
  return COLOUR_TOKENS.map((token) => {
    const value = valueOf(vars, token.name)
    const raw = raws[token.name]
    return { ...token, value, note: raw === undefined ? null : variantNote(raw, value, groundName) }
  })
}

export interface ContrastCheck {
  /** "text / bg": which pair, in the style sheet's short names. */
  readonly label: string
  readonly foreground: string
  readonly background: string
  readonly ratio: number
  readonly passes: boolean
}

export function contrastCheck(label: string, foreground: string, background: string): ContrastCheck {
  const ratio = contrastRatio(foreground, background)
  return { label, foreground, background, ratio, passes: ratio >= AA_TEXT }
}

/** The live contrast ratios: text, secondary text and button ink, plus the readable variants. */
export function contrastChecks(look: Look): readonly ContrastCheck[] {
  const ground = GROUNDS[look.ground]
  const accent = ACCENTS[look.accent]
  const vars = lookToCssVars(look)
  return [
    contrastCheck('text / bg', ground.text, ground.bg),
    contrastCheck('text-2 / bg', ground.text2, ground.bg),
    contrastCheck('text / surface', ground.text, ground.surface),
    contrastCheck('text-2 / surface', ground.text2, ground.surface),
    contrastCheck(`ink / ${accent.name.toLowerCase()}`, accent.ink, accent.hex),
    contrastCheck('accent text / bg', valueOf(vars, '--pl-accent-text'), ground.bg),
    contrastCheck('error text / bg', valueOf(vars, '--pl-error-text'), ground.bg),
  ]
}

/** "16.36:1". */
export function formatRatio(ratio: number): string {
  return `${ratio.toFixed(2)}:1`
}

/** The active look's variables as a block to paste into a stylesheet ("Copy tokens"). */
export function rootBlock(look: Look): string {
  const lines = Object.entries(lookToCssVars(look)).map(([name, value]) => `  ${name}: ${value};`)
  return [':root {', ...lines, '}'].join('\n')
}

export interface Specimen {
  readonly label: string
  readonly sample: string
}

/** "Editorial · Instrument Serif italic 400": the type style, its face, style and weight. */
export function displayLine(look: Look): string {
  const font = TYPE_STYLES[look.font]
  return `${font.name} · ${font.face} ${font.style} ${font.weight}`
}

/** The display specimens at the look's own sizes: hero, status and title. */
export function displaySpecimens(look: Look): readonly Specimen[] {
  const { sizes } = TYPE_STYLES[look.font]
  return [
    { label: `Hero ${sizes.hero}px`, sample: 'The fun way.' },
    { label: `Status ${sizes.status}px`, sample: 'Searching.' },
    { label: `Title ${sizes.card}px`, sample: 'Laptop shortlist' },
  ]
}

export interface TextSpecimen extends Specimen {
  /** The app/play-theme.css token it is set in. */
  readonly token: string
}

/** Reading type is Geist in every look. Labels match app/play-theme.css (lib/looks/__tests__/style.test.ts). */
export const TEXT_SPECIMENS: readonly TextSpecimen[] = [
  { token: '--pl-text-reply', label: 'Reply 18 / 1.6', sample: 'Three laptops make the cut under £700.' },
  { token: '--pl-text-body', label: 'Body 16 / 1.6', sample: 'Can you make me a shortlist?' },
  { token: '--pl-text-button-sm', label: 'Button 14 / 600', sample: 'Try again' },
  { token: '--pl-text-label', label: 'Label 11 caps', sample: 'What Auto did' },
  { token: '--pl-text-mono', label: 'Mono 13', sample: '✓ Searched the web · 3 results · 2.1 s' },
]

export interface ScaleStep {
  readonly token: string
  readonly px: number
}

/** The 4px spacing scale in app/play-theme.css. */
export const SPACING_STEPS: readonly ScaleStep[] = [
  { token: '--pl-space-1', px: 4 },
  { token: '--pl-space-2', px: 8 },
  { token: '--pl-space-3', px: 12 },
  { token: '--pl-space-4', px: 16 },
  { token: '--pl-space-5', px: 24 },
  { token: '--pl-space-6', px: 40 },
  { token: '--pl-space-7', px: 56 },
]

export type RadiusKind = 'paper' | 'tool' | 'button' | 'card' | 'pill'

export interface RadiusStep {
  readonly kind: RadiusKind
  readonly label: string
}

/** The five Ember radii (--pl-radius-<kind> in app/play-theme.css). */
export const RADII: readonly RadiusStep[] = [
  { kind: 'paper', label: '6 paper' },
  { kind: 'tool', label: '8 tool' },
  { kind: 'button', label: '10 button' },
  { kind: 'card', label: '14 card' },
  { kind: 'pill', label: 'pill chip' },
]

/** The token a radius is drawn with. */
export function radiusToken(kind: RadiusKind): string {
  return `--pl-radius-${kind}`
}

/** The Ember motion notes, with the theme's own numbers (app/play-theme.css: Motion). */
export const MOTION_NOTES: readonly string[] = [
  'Message arrives: rise 12px and fade, 180ms ease-out',
  'Ring: scale 1 → 1.06 with glow, 2.4s loop',
  'Page glow follows Auto: accent when busy or done, coral on a snag, off asleep (600ms crossfade)',
  'Deliverable lands: drop 16px, overshoot 1.04, 420ms',
  'Every loop stops under reduced motion',
]

/** When Auto is in each state (the Ember style sheet's mascot slot captions). */
const MASCOT_WHEN: Readonly<Record<MascotName, string>> = {
  hello: 'First open',
  idle: 'Nothing happening',
  listening: 'You are typing',
  thinking: 'Waiting for a reply',
  working: 'A tool is running',
  searching: 'Web search',
  reading: 'Reading documents',
  celebrate: 'Deliverable ready',
  oops: 'Error or needs help',
  sleeping: 'Automatos not running',
  lost: '404',
  loading: 'Loading',
}

export interface MascotSlot {
  readonly state: MascotName
  readonly when: string
  readonly tone: RingTone
  /** The ring's #rrggbb colour under this look (lib/mascot/ring.ts). */
  readonly ring: string
}

/** Auto's 12 states with the ring colour each takes under the look. */
export function mascotSlots(look: Look): readonly MascotSlot[] {
  return MASCOT_STATES.map((state) => ({
    state,
    when: MASCOT_WHEN[state],
    tone: ringTone(state),
    ring: ringColour(state, look),
  }))
}
