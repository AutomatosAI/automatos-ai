/**
 * PRD-261 A-19d2 — replaces lib/looks/__tests__/prepaint.test.ts (the core ledger's
 * coreReplacement, A-19f2): run as the browser runs it, in the page, the boot script
 * (THEME_BOOT_SCRIPT) puts the stored look's variables AND Studio's tokens on <html> only under an
 * installed theme with looks, from the theme's own prepaint table; a stored value that is not a
 * valid look gets the default look; the per-account pointer picks the scope (F5). And the generic
 * look runtime's paint() is that walk as a function: on hello-a09b's table the boot script and
 * paint() give identical variables, so what the runtime paints after hydration is what the first
 * frame showed. Studio's tokens are the table's own (precomputed by the theme), never computed here.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { hexToTriplet } from '@/lib/look-kit/hsl'
import { lookScope } from '@/lib/look-kit/scope'
import { baseThemeBoot, builtinThemeBoot, type ThemeBoot } from '@/lib/theme-host/boot'
import type { PrepaintKnobTable, PrepaintTable } from '@/lib/theme-host/boot-types'
import { applyPaint, clearPaint, paint, paintNames } from '@/lib/theme-host/looks/paint'
import { lookKeys } from '@/lib/theme-host/looks/storage'
import type { UiStyle } from '@/lib/theme-host/registry-types'

import { cookieReader, runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'
import { DUSK, HELLO, HELLO_RULES, helloThemeBoot, INK, myLook, PAPER } from './hello-looks'

const TABLE = HELLO_RULES.table
const INSTALL_STYLE: UiStyle = 'theme:install:0b7e4c1a-9f3d-4c2e-8a51-3d2f6b7a9c10'
const MINE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a')
const THEIRS = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_b')
/** Studio's tokens a theme's table carries per ground (tailwind reads hsl(var(--x))). */
const STUDIO_BACKGROUND = '--background'
const STUDIO_FOREGROUND = '--foreground'
const OTHER_INLINE = 'outline-color'

function root(): HTMLElement {
  return document.documentElement
}

/** Every variable the table can paint that is set on <html> now, trimmed. */
function painted(table: PrepaintTable = TABLE): Readonly<Record<string, string>> {
  const entries = paintNames(table).map((name): [string, string] => [name, root().style.getPropertyValue(name).trim()])
  return Object.fromEntries(entries.filter(([, value]) => value !== ''))
}

function storeActive(scope: string | null, value: unknown): void {
  const key = scope === null ? HELLO.keys.active : lookKeys(scope, HELLO.keys).active
  window.localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value))
}

function pointAt(scope: string): void {
  window.localStorage.setItem(HELLO.keys.pointer, scope)
}

/** hello's table with Studio's tokens on each ground, as a theme precomputes them into its table. */
function withStudioTokens(table: PrepaintTable): PrepaintTable {
  const ground: PrepaintKnobTable = Object.fromEntries(
    Object.entries(table.knobs.ground).map(([key, vars]) => [
      key,
      { ...vars, [STUDIO_BACKGROUND]: hexToTriplet(vars['--a09b-bg']), [STUDIO_FOREGROUND]: hexToTriplet(vars['--a09b-ink']) },
    ]),
  )
  return { ...table, knobs: { ...table.knobs, ground } }
}

function resetRoot(): void {
  root().removeAttribute('style')
  root().className = ''
}

beforeEach(() => {
  window.localStorage.clear()
  resetRoot()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
  resetRoot()
})

const SAMPLES = [PAPER, DUSK, INK, myLook(), myLook({ glow: 0 }), myLook({ glow: 40, ground: 'ink' }), myLook({ accent: 'blue', ground: 'paper', glow: 33 })]

describe('the boot script and paint() on the hello table', () => {
  it.each(SAMPLES.map((look) => [look.id, look] as const))('give identical variables for %s', (_id, look) => {
    storeActive(MINE, look)
    pointAt(MINE)
    runThemeBoot(helloThemeBoot())
    expect(painted()).toEqual(paint(TABLE, look))
    resetRoot()
    applyPaint(root(), TABLE, look)
    expect(painted()).toEqual(paint(TABLE, look))
  })

  it('paint() names only what the table can paint, and every name it can paint is in some look', () => {
    const names = new Set(paintNames(TABLE))
    SAMPLES.forEach((look) => Object.keys(paint(TABLE, look)).forEach((name) => expect(names.has(name)).toBe(true)))
    const seen = new Set(SAMPLES.flatMap((look) => Object.keys(paint(TABLE, look))))
    expect([...names].filter((name) => !seen.has(name))).toEqual([])
  })
})

describe('the boot script under an installed theme with looks', () => {
  it("puts the stored look's variables and Studio's tokens on <html>, from the theme's table", () => {
    const table = withStudioTokens(TABLE)
    storeActive(MINE, DUSK)
    pointAt(MINE)
    runThemeBoot(helloThemeBoot({ prepaint: table }))
    expect(painted(table)).toEqual(paint(table, DUSK))
    expect(root().style.getPropertyValue(STUDIO_BACKGROUND)).toBe(hexToTriplet(TABLE.knobs.ground.dusk['--a09b-bg']))
  })

  it.each<UiStyle>(['studio', 'classic'])("sets nothing on <html> under %s's boot", (style) => {
    storeActive(MINE, DUSK)
    pointAt(MINE)
    runThemeBoot(builtinThemeBoot(style, cookieReader()))
    expect(root().style.length).toBe(0)
  })

  it("sets nothing on <html> under an installed theme's base boot, or one with no table", () => {
    storeActive(null, DUSK)
    const bases: ThemeBoot[] = [baseThemeBoot(INSTALL_STYLE), helloThemeBoot({ prepaint: null })]
    bases.forEach((boot) => {
      runThemeBoot(boot)
      expect(root().style.length).toBe(0)
    })
  })
})

describe('a stored value that is not a valid look', () => {
  const tampered: readonly unknown[] = [
    '{not json',
    'null',
    '"dusk"',
    '[]',
    { ...DUSK, ground: 'graphite' },
    { ...DUSK, ground: '__proto__' },
    { ...DUSK, accent: 'toString' },
    { ...DUSK, accent: 'green;background:url(x)' },
    { ...DUSK, glow: 41 },
    { ...DUSK, glow: -1 },
    { ...DUSK, glow: 2.5 },
    { ...DUSK, glow: '24' },
    { ...DUSK, schema: 'automatos-e2e-a09b-look/v0' },
    { ...DUSK, interface: 'studio' },
    JSON.stringify(DUSK).replace('"ground":"dusk"', '"__proto__":{"ground":"dusk"}'),
  ]

  it.each(tampered.map((value) => [JSON.stringify(value), value] as const))('gets the default look: %s', (_label, value) => {
    storeActive(MINE, value)
    pointAt(MINE)
    runThemeBoot(helloThemeBoot())
    expect(painted()).toEqual(paint(TABLE, PAPER))
  })

  it("paint() falls back to the table's default look the same way", () => {
    expect(paint(TABLE, { ...DUSK, ground: 'graphite' })).toEqual(paint(TABLE, TABLE.defaults))
    expect(paint(TABLE, TABLE.defaults)).toEqual(paint(TABLE, PAPER))
  })

  it('gets the default look, without throwing, when localStorage refuses to be read', () => {
    storeActive(MINE, DUSK)
    pointAt(MINE)
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    })
    expect(() => runThemeBoot(helloThemeBoot())).not.toThrow()
    expect(painted()).toEqual(paint(TABLE, PAPER))
  })
})

describe('the per-account pointer picks the scope (F5)', () => {
  it("paints the active look of the account the pointer names, not another account's or the unscoped one", () => {
    storeActive(MINE, INK)
    storeActive(THEIRS, DUSK)
    storeActive(null, myLook())
    pointAt(MINE)
    runThemeBoot(helloThemeBoot())
    expect(painted()).toEqual(paint(TABLE, INK))
  })

  it('paints the default look when the account the pointer names has no active look yet', () => {
    storeActive(THEIRS, DUSK)
    pointAt(MINE)
    runThemeBoot(helloThemeBoot())
    expect(painted()).toEqual(paint(TABLE, PAPER))
  })

  it('paints the unscoped look when no pointer is set', () => {
    storeActive(null, INK)
    runThemeBoot(helloThemeBoot())
    expect(painted()).toEqual(paint(TABLE, INK))
  })
})

describe('the runtime on <html>', () => {
  const table: PrepaintTable = {
    knobs: { tint: { warm: { '--sample-a': '#ff6a1f', '--sample-b': '#101010' }, cold: { '--sample-a': '#1f6aff' } } },
    pairs: [],
    numeric: [],
    defaults: { tint: 'warm' },
  }

  it("leaves nothing of the previous look: a variable only the last one set comes off", () => {
    applyPaint(root(), table, { tint: 'warm' })
    applyPaint(root(), table, { tint: 'cold' })
    expect(painted(table)).toEqual({ '--sample-a': '#1f6aff' })
  })

  it('takes every variable the table can paint off <html>, and no other inline style', () => {
    root().style.setProperty(OTHER_INLINE, 'red')
    applyPaint(root(), table, { tint: 'warm' })
    clearPaint(root(), table)
    expect(painted(table)).toEqual({})
    expect(root().style.getPropertyValue(OTHER_INLINE)).toBe('red')
  })
})
