// @vitest-environment node
/**
 * PRD-259 S-010b, PRD-260 P260-FX-05 (M41, M107) — `dark:` classes follow the tone.
 *
 * Since S-010a (`darkMode: ['class', '.dark']`, cherry-picked by PRD-260 D9) every `dark:`
 * class renders, in Classic Dark, Studio Dark and a skin's dark tone alike. Three rules:
 *
 * 1. No fixed palette colour. A neutral (`dark:text-gray-200`, `dark:bg-white`), a palette
 *    hue (`dark:text-amber-500`, `dark:bg-blue-950/20`) or an arbitrary colour
 *    (`dark:border-[#333]`, `dark:bg-[rgb(0,0,0)]`) ignores the tone's tokens and paints
 *    over Studio and every skin. Use the token class (`text-foreground`, `bg-card`,
 *    `border-border`, `text-warning`, …), or drop the `dark:` class when the base class is
 *    already the token.
 * 2. No faded token text. `dark:text-<token>/<n>` under 90% sits below WCAG AA on its own
 *    tint (`dark:text-info/80` on `dark:bg-info/20` is about 3:1). Use `dark:text-foreground`
 *    and let the tint carry the colour; dark-tone-contrast.test.ts holds the ratios.
 * 3. No cancelled hover or focus. Tailwind emits the dark variant after hover:/focus: at the
 *    same specificity, so a bare `dark:bg-…` in the same class list as `hover:bg-…` wins
 *    in every dark tone and the hover never shows. Pair it with `dark:hover:bg-…`, or drop it.
 *
 * docs/PRDS/PRD-259-DARK-VARIANT-AUDIT.md has the verdict rules and one row per use.
 */
import { describe, it, expect } from 'vitest'

import { classLists, scannedSources } from './class-lists'

const PALETTE_HUES = [
  'red', 'orange', 'amber', 'yellow', 'lime', 'green', 'emerald', 'teal', 'cyan', 'sky', 'blue',
  'indigo', 'violet', 'purple', 'fuchsia', 'pink', 'rose', 'gray', 'slate', 'zinc', 'neutral', 'stone',
]
const COLOUR_UTILITY = String.raw`(?:bg|text|border(?:-[xytrbls])?|ring-offset|ring|from|to|via|divide|outline|fill|stroke|placeholder|decoration|shadow|accent|caret)`
/** A literal colour in brackets: a hex or a colour function of numbers, not hsl(var(--token)). */
const ARBITRARY_COLOUR = String.raw`\[(?:#[0-9a-fA-F]+|(?:rgba?|hsla?|oklch|oklab|lab|lch)\(\s*[-\d.])[^\]]*\]`

/** `dark:` (any variant chain after it) + a colour utility + a palette colour, white, black or an arbitrary colour. */
const FIXED_DARK_CLASS = new RegExp(
  String.raw`dark:(?:[a-z-]+:)*` +
    `${COLOUR_UTILITY}-` +
    String.raw`(?:(?:${PALETTE_HUES.join('|')})-\d+|white\b|black\b|${ARBITRARY_COLOUR})` +
    String.raw`[^\s"'\`]*`,
  'g',
)

/** `dark:text-<token>/<n>`: a token colour at a fixed opacity (palette shades never match: no digits). */
const FADED_DARK_TEXT = /dark:(?:[a-z-]+:)*text-[a-z]+(?:-[a-z]+)*\/(\d+)(?![\w.])/g
const MIN_DARK_TEXT_OPACITY = 90

const COLOUR_PROPS = 'bg|text|border|ring|outline|divide|fill|stroke|shadow|decoration|from|via|to|placeholder|accent|caret'
const STATES = 'hover|focus|focus-visible|focus-within|active'
const BARE_DARK = new RegExp(`^dark:(${COLOUR_PROPS})-`)
const STATE_CLASS = new RegExp(`^(${STATES}):(${COLOUR_PROPS})-`)
const DARK_STATE_CLASS = new RegExp(`^(?:dark:(${STATES})|(${STATES}):dark):(${COLOUR_PROPS})-`)

const SHEET_ARTIFACT_RATCHET = 'fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines)'
const MESSAGE_RATCHET = 'fix — blocked by the code-shape ratchet (Message, 409 lines)'
const POWER_UP_CARD_RATCHET = 'fix — blocked by the code-shape ratchet (PowerUpCard, 222 lines)'

/**
 * file:class pairs the audit could not fix: each sits inside a function over the
 * AGENTS.md limit, where the changed-lines gate fails on any touched line. Remove an
 * entry when its function is split and the class moves to a token.
 */
const ALLOWLIST: Record<string, string> = {
  'components/chatbot/sheet-artifact.tsx dark:border-gray-800/60': SHEET_ARTIFACT_RATCHET,
  'components/chatbot/sheet-artifact.tsx dark:text-gray-200': SHEET_ARTIFACT_RATCHET,
  'components/chatbot/sheet-artifact.tsx dark:text-gray-100': SHEET_ARTIFACT_RATCHET,
  'components/chatbot/sheet-artifact.tsx dark:divide-gray-800/70': SHEET_ARTIFACT_RATCHET,
  'components/chatbot/message.tsx dark:text-gray-200': MESSAGE_RATCHET,
  'components/onboarding/power-up-card.tsx dark:text-amber-500': POWER_UP_CARD_RATCHET,
  'components/onboarding/power-up-card.tsx dark:text-emerald-400': POWER_UP_CARD_RATCHET,
}

/** Parsing every file that carries a dark: class takes a moment; the default 5 s is tight on a busy runner. */
const SCAN_TIMEOUT_MS = 120_000

function fixedDarkClasses(source: string): string[] {
  return [...source.matchAll(FIXED_DARK_CLASS)].map((match) => match[0])
}

function fadedDarkText(source: string): string[] {
  return [...source.matchAll(FADED_DARK_TEXT)]
    .filter((match) => Number(match[1]) < MIN_DARK_TEXT_OPACITY)
    .map((match) => match[0])
}

/** `<bare dark: class> cancels <state class>` for each hover/focus class a bare dark: class overrides. */
function cancelledStates(classes: readonly string[]): string[] {
  const bare = new Map<string, string>()
  const cleared = new Set<string>()
  for (const cls of classes) {
    const dark = BARE_DARK.exec(cls)
    if (dark) bare.set(dark[1], cls)
    const darkState = DARK_STATE_CLASS.exec(cls)
    if (darkState) cleared.add(`${darkState[1] ?? darkState[2]}:${darkState[3]}`)
  }
  return classes.flatMap((cls) => {
    const state = STATE_CLASS.exec(cls)
    if (!state) return []
    const darkClass = bare.get(state[2])
    return darkClass && !cleared.has(`${state[1]}:${state[2]}`) ? [`${darkClass} cancels ${cls}`] : []
  })
}

function cancelledInSource(source: string, fileName: string): string[] {
  return [...new Set(classLists(source, fileName).flatMap(cancelledStates))]
}

/** Every `<file> <class>` pair in the scanned tree that `find` reports. */
function foundPairs(find: (source: string) => string[]): string[] {
  return scannedSources().flatMap(({ rel, text }) => find(text).map((cls) => `${rel} ${cls}`))
}

describe('PRD-259 S-010b — dark: classes use tokens, not fixed colours', () => {
  it('the pattern catches fixed neutrals and hexes, and passes token classes', () => {
    const caught = fixedDarkClasses(
      'dark:text-gray-200 dark:hover:bg-zinc-800/50 dark:border-[#333333] dark:text-white dark:bg-black/40',
    )
    expect(caught).toEqual([
      'dark:text-gray-200',
      'dark:hover:bg-zinc-800/50',
      'dark:border-[#333333]',
      'dark:text-white',
      'dark:bg-black/40',
    ])
    expect(fixedDarkClasses('dark:text-foreground dark:bg-background/40 dark:border-border dark:text-info/80')).toEqual([])
  })

  it('M107: the pattern catches every palette hue and arbitrary colour, and still passes token classes', () => {
    const caught = fixedDarkClasses(
      'dark:text-amber-500 dark:bg-[rgb(0,0,0)] dark:bg-blue-950/20 dark:text-emerald-400 dark:border-t-rose-300 dark:ring-offset-sky-900',
    )
    expect(caught).toEqual([
      'dark:text-amber-500',
      'dark:bg-[rgb(0,0,0)]',
      'dark:bg-blue-950/20',
      'dark:text-emerald-400',
      'dark:border-t-rose-300',
      'dark:ring-offset-sky-900',
    ])
    expect(fixedDarkClasses('dark:text-warning dark:bg-info/20 dark:bg-[hsl(var(--card))] dark:text-destructive')).toEqual([])
  })

  it('no dark: class outside the allowlist paints a fixed palette colour', () => {
    const offenders = [...new Set(foundPairs(fixedDarkClasses))].filter((pair) => !(pair in ALLOWLIST))
    expect(offenders).toEqual([])
  })

  it('every allowlist entry still exists (drop it once its class is fixed)', () => {
    const found = new Set(foundPairs(fixedDarkClasses))
    const stale = Object.keys(ALLOWLIST).filter((pair) => !found.has(pair))
    expect(stale).toEqual([])
  })
})

describe('PRD-260 P260-FX-05 (M41, M107) — dark: text stays readable and hover stays visible', () => {
  it('the faded-text pattern catches token text under 90% and passes full or 90% text', () => {
    expect(fadedDarkText('dark:text-info/80 dark:text-destructive/70 dark:group-hover:text-agent/50')).toEqual([
      'dark:text-info/80',
      'dark:text-destructive/70',
      'dark:group-hover:text-agent/50',
    ])
    expect(fadedDarkText('dark:text-foreground dark:text-foreground/90 dark:bg-info/20 text-info/70 dark:text-gray-200')).toEqual([])
  })

  it('no dark: text class fades a token colour below 90%', () => {
    expect([...new Set(foundPairs(fadedDarkText))]).toEqual([])
  })

  it('a bare dark: class beside hover:/focus: of the same property is caught, unless dark:hover: restores it', () => {
    expect(cancelledStates(['hover:bg-warning/5', 'hover:border-warning/40', 'dark:bg-background/40', 'dark:border-border'])).toEqual([
      'dark:bg-background/40 cancels hover:bg-warning/5',
      'dark:border-border cancels hover:border-warning/40',
    ])
    expect(cancelledStates(['hover:bg-muted/50', 'focus:bg-muted/50', 'dark:bg-info/10'])).toEqual([
      'dark:bg-info/10 cancels hover:bg-muted/50',
      'dark:bg-info/10 cancels focus:bg-muted/50',
    ])
    expect(cancelledStates(['hover:bg-secondary/40', 'dark:bg-background/30', 'dark:hover:bg-background/60'])).toEqual([])
    expect(cancelledStates(['group-hover:text-foreground/80', 'dark:text-foreground/90', 'hover:bg-muted'])).toEqual([])
  })

  it('one class list is a className attribute or a cn() call, read as code: comments and JSX text never count', () => {
    const joined = "const a = <button className={cn('px-3 hover:bg-muted/50', isRunning && 'dark:bg-info/10')} />"
    expect(cancelledInSource(joined, 'probe.tsx')).toEqual(['dark:bg-info/10 cancels hover:bg-muted/50'])
    const prose = '// hover:bg-muted dark:bg-card\nconst b = <p>hover:bg-muted dark:bg-card</p>'
    expect(cancelledInSource(prose, 'probe.tsx')).toEqual([])
  })

  it(
    'no class list in the tree cancels its own hover or focus in dark tones',
    () => {
      const offenders = scannedSources()
        .filter(({ text }) => text.includes('dark:'))
        .flatMap(({ rel, text }) => cancelledInSource(text, rel).map((found) => `${rel}: ${found}`))
      expect(offenders).toEqual([])
    },
    SCAN_TIMEOUT_MS,
  )
})
