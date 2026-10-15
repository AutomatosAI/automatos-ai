// @vitest-environment node
/**
 * PRD-260 P260-FX-05 (M41) — text on a tint clears WCAG AA in the dark tones.
 *
 * The now-live dark: classes faded token text on its own tint: `dark:text-info/80` on
 * `dark:bg-info/20` is about 3:1 in Classic Dark. The fix puts `dark:text-foreground` on
 * the tint and lets the tint carry the colour. This reads Classic Dark (`:root` + `.dark`)
 * and Studio Dark (+ `.studio` + `.studio.dark`) from app/globals.css, in cascade order,
 * and checks the ratios on --card:
 *
 * (a) foreground over destructive, info and agent at 20%, and text-info over info/10
 *     (the model chip once its faded dark: class is gone), are at least 4.5:1;
 * (b) every class string in the tree that puts a dark:text-<token> on a token tint (its
 *     dark:bg-<token>, else its bg-<token>) is at least 4.5:1 in both tones.
 */
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import { FRONTEND, literalClassLists, scannedSources } from './class-lists'
import { cssRules } from './css-cascade'

const AA_TEXT = 4.5
const GLOBALS_CSS = path.join(FRONTEND, 'app/globals.css')
/** The rules each tone's <html> matches, lowest precedence first (specificity, then source order). */
const TONES: Record<string, readonly string[]> = {
  'Classic Dark': [':root', '.dark'],
  'Studio Dark': [':root', '.dark', '.studio', '.studio.dark'],
}
const GROUND = 'card'
const TINTS = ['destructive', 'info', 'agent']
const HSL_TRIPLET = /^(-?[\d.]+)\s+([\d.]+)%\s+([\d.]+)%$/
const TOKEN_COLOUR = /^([a-z]+(?:-[a-z]+)*)(?:\/(\d+))?$/
const PERCENT = 100
const DEGREES = 360
/** WCAG 2.x relative luminance: the sRGB transfer function, then the channel weights. */
const SRGB = { threshold: 0.04045, slope: 12.92, offset: 0.055, scale: 1.055, gamma: 2.4 }
const LUMA_WEIGHTS: readonly number[] = [0.2126, 0.7152, 0.0722]
const FLARE = 0.05

type Rgb = readonly [number, number, number]
type Tone = ReadonlyMap<string, Rgb>

function hslToRgb(h: number, s: number, l: number): Rgb {
  const sat = s / PERCENT
  const light = l / PERCENT
  const k = (n: number) => (n + h / (DEGREES / 12)) % 12
  const a = sat * Math.min(light, 1 - light)
  const channel = (n: number) => light - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1))
  return [channel(0), channel(8), channel(4)]
}

/** The HSL triplet custom properties the given rules declare, later ones winning. */
function toneTokens(css: string, selectors: readonly string[]): Tone {
  const rules = cssRules(css)
  const tokens = new Map<string, Rgb>()
  for (const selector of selectors) {
    for (const rule of rules.filter((r) => r.selectors.includes(selector))) {
      for (const declaration of rule.body.split(';')) {
        const colon = declaration.indexOf(':')
        const name = declaration.slice(0, colon).trim()
        const triplet = HSL_TRIPLET.exec(declaration.slice(colon + 1).trim())
        if (colon > 0 && name.startsWith('--') && triplet) {
          tokens.set(name.slice(2), hslToRgb(Number(triplet[1]), Number(triplet[2]), Number(triplet[3])))
        }
      }
    }
  }
  return tokens
}

/** `info/20` → the tone's info at 20% opacity; null when the name is not one of the tone's tokens. */
function paint(tone: Tone, colour: string): { rgb: Rgb; alpha: number } | null {
  const match = TOKEN_COLOUR.exec(colour)
  const rgb = match ? tone.get(match[1]) : undefined
  if (!match || !rgb) return null
  return { rgb, alpha: match[2] ? Number(match[2]) / PERCENT : 1 }
}

function over(top: { rgb: Rgb; alpha: number }, bottom: Rgb): Rgb {
  const mix = (i: 0 | 1 | 2) => top.alpha * top.rgb[i] + (1 - top.alpha) * bottom[i]
  return [mix(0), mix(1), mix(2)]
}

function luminance(rgb: Rgb): number {
  const linear = rgb.map((c) => (c <= SRGB.threshold ? c / SRGB.slope : ((c + SRGB.offset) / SRGB.scale) ** SRGB.gamma))
  return LUMA_WEIGHTS.reduce((sum, weight, i) => sum + weight * linear[i], 0)
}

/** WCAG contrast of `text` over `tint`, the tint laid on --card; null when either is not a token. */
function contrastOnTint(tone: Tone, text: string, tint: string): number | null {
  const card = tone.get(GROUND)
  const textPaint = paint(tone, text)
  const tintPaint = paint(tone, tint)
  if (!card || !textPaint || !tintPaint) return null
  const ground = over(tintPaint, card)
  const [hi, lo] = [luminance(over(textPaint, ground)), luminance(ground)].sort((x, y) => y - x)
  return (hi + FLARE) / (lo + FLARE)
}

/** A class string's dark text and the tint under it: `dark:text-x` on `dark:bg-y`, else on `bg-y`. */
function tintedDarkText(classes: readonly string[]): { text: string; tint: string } | null {
  const after = (prefix: string) => classes.find((cls) => cls.startsWith(prefix))?.slice(prefix.length)
  const text = after('dark:text-')
  const tint = after('dark:bg-') ?? after('bg-')
  return text && tint ? { text, tint } : null
}

const css = fs.readFileSync(GLOBALS_CSS, 'utf8')
const tones = Object.entries(TONES).map(([name, selectors]) => ({ name, tone: toneTokens(css, selectors) }))

describe('PRD-260 P260-FX-05 (M41) — dark-tone text on a tint clears AA', () => {
  it('reads both dark tones from globals.css', () => {
    for (const { name, tone } of tones) {
      for (const token of ['foreground', GROUND, ...TINTS]) expect(tone.has(token), `${name} --${token}`).toBe(true)
    }
    const classic = tones[0].tone.get('foreground')
    const studio = tones[1].tone.get('foreground')
    expect(studio, 'Studio Dark has its own cream foreground').not.toEqual(classic)
  })

  it.each(tones)('$name: foreground over a 20% tint, and the info chip, are at least 4.5:1', ({ name, tone }) => {
    for (const tint of TINTS) {
      expect(contrastOnTint(tone, 'foreground', `${tint}/20`), `${name}: foreground on ${tint}/20`).toBeGreaterThanOrEqual(AA_TEXT)
    }
    expect(contrastOnTint(tone, 'info', 'info/10'), `${name}: text-info on bg-info/10`).toBeGreaterThanOrEqual(AA_TEXT)
  })

  it('the old faded pairs were below AA (why the fix exists)', () => {
    for (const { name, tone } of tones) {
      expect(contrastOnTint(tone, 'info/80', 'info/20'), `${name}: info/80 on info/20`).toBeLessThan(AA_TEXT)
      expect(contrastOnTint(tone, 'destructive/80', 'destructive/20'), `${name}: destructive/80 on destructive/20`).toBeLessThan(AA_TEXT)
    }
  })

  it('every dark:text class on a token tint in the tree clears AA in both dark tones', () => {
    const below = scannedSources()
      .filter(({ text }) => text.includes('dark:text-'))
      .flatMap(({ rel, text }) =>
        literalClassLists(text, rel).flatMap((classes) => {
          const pair = tintedDarkText(classes)
          if (!pair) return []
          return tones.flatMap(({ name, tone }) => {
            const ratio = contrastOnTint(tone, pair.text, pair.tint)
            return ratio !== null && ratio < AA_TEXT
              ? [`${rel}: dark:text-${pair.text} on ${pair.tint} is ${ratio.toFixed(2)}:1 in ${name}`]
              : []
          })
        }),
      )
    expect(below).toEqual([])
  })
})
