/**
 * PRD-261 A-01 — the spike's theme.json and look preset are checked before anything is applied:
 * the core's own <html> classes are refused, every file must resolve under the theme's folder on
 * this origin, and a preset value is a colour, a length or a number.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { RESERVED_HTML_CLASSES, THEME_SPIKE_ROOT } from '../constants'
import { parseLookPreset, parseSpikeManifest, resolveUnderRoot } from '../manifest'
import { STYLE_HTML_CLASS_NAMES } from '@/lib/ui-style'

/** The hello theme's manifest, as the browser lane serves it. */
const FIXTURES = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-theme')
const HELLO_MANIFEST = JSON.parse(readFileSync(path.join(FIXTURES, 'theme.json'), 'utf8')) as Readonly<Record<string, unknown>>
const MISMATCHED_MANIFEST: unknown = JSON.parse(readFileSync(path.join(FIXTURES, 'theme.engines-mismatch.json'), 'utf8'))

function withField(field: string, value: unknown): Record<string, unknown> {
  return { ...HELLO_MANIFEST, [field]: value }
}

function detailOf(raw: unknown): string | null {
  const parsed = parseSpikeManifest(raw)
  return parsed.ok ? null : parsed.detail
}

describe('PRD-261 A-01 — parseSpikeManifest', () => {
  it("reads the hello theme's manifest, with its files resolved under the theme's folder", () => {
    const parsed = parseSpikeManifest(HELLO_MANIFEST)
    expect(parsed.ok).toBe(true)
    if (!parsed.ok) return
    const root = `${window.location.origin}${THEME_SPIKE_ROOT}`
    expect(parsed.manifest).toMatchObject({ id: 'automatos-e2e/hello', htmlClass: 'hello-spike', sdkRange: '^1.0.0', slots: ['frame@1'] })
    expect(parsed.manifest.entry).toBe(`${root}entry.js`)
    expect(parsed.manifest.styleSheet).toBe(`${root}theme.css`)
    expect(parsed.manifest.preset).toBe(`${root}presets/default.look.json`)
  })

  it('reads the engines-mismatch manifest too: the range is checked by the loader, not here', () => {
    const parsed = parseSpikeManifest(MISMATCHED_MANIFEST)
    expect(parsed.ok && parsed.manifest.sdkRange).toBe('^2.0.0')
  })

  it("refuses every class the core owns on <html> as the theme's class", () => {
    expect(RESERVED_HTML_CLASSES).toEqual(expect.arrayContaining([...STYLE_HTML_CLASS_NAMES, 'classic', 'dark', 'light']))
    for (const reserved of RESERVED_HTML_CLASSES) {
      expect(detailOf(withField('htmlClass', reserved)), reserved).not.toBeNull()
    }
  })

  it.each(['Hello', 'hi', 'hello spike', 'hello_spike', '1hello', 'a'.repeat(33), 42])('refuses the htmlClass %j', (htmlClass) => {
    expect(detailOf(withField('htmlClass', htmlClass))).not.toBeNull()
  })

  it.each(['../entry.js', '/entry.js', '//evil.example/x.js', 'https://evil.example/x.js', 'entry.js?x=:', 'a/../../b.js', 'a\\b.js', '', 7])(
    'refuses the entry %j',
    (entry) => {
      expect(detailOf(withField('entry', entry))).not.toBeNull()
    },
  )

  it('refuses an unknown slot, a missing frame slot, a wrong schema, base or engines', () => {
    expect(detailOf(withField('slots', ['frame@1', 'sidebar@1']))).toContain('sidebar@1')
    expect(detailOf(withField('slots', []))).not.toBeNull()
    expect(detailOf(withField('schema', 'automatos-theme/v2'))).not.toBeNull()
    expect(detailOf(withField('base', 'classic'))).not.toBeNull()
    expect(detailOf(withField('engines', {}))).not.toBeNull()
    expect(detailOf(withField('id', ''))).not.toBeNull()
    expect(detailOf(null)).not.toBeNull()
    expect(detailOf([HELLO_MANIFEST])).not.toBeNull()
  })
})

describe('PRD-261 A-01 — resolveUnderRoot', () => {
  it("resolves a plain relative path under the theme's folder on this origin", () => {
    expect(resolveUnderRoot('presets/default.look.json')).toBe(`${window.location.origin}${THEME_SPIKE_ROOT}presets/default.look.json`)
    expect(resolveUnderRoot('C:/x.js')).toBeNull()
    expect(resolveUnderRoot('entry.js#x')).toBeNull()
  })
})

describe('PRD-261 A-01 — parseLookPreset', () => {
  it.each(['#2f6fed', '#fff', '12px', '1.5rem', '50%', '0', 'rgb(1, 2, 3)', 'hsla(20, 8%, 7%, 0.5)'])('accepts the value %j', (value) => {
    expect(parseLookPreset({ vars: { '--hello-accent': value } }).ok).toBe(true)
  })

  it.each(['url(x)', 'red;}', 'expression(1)', 'red', '#2f6fed;', 'rgb(1,2,3);x', '', 12])('refuses the value %j', (value) => {
    expect(parseLookPreset({ vars: { '--hello-accent': value } }).ok).toBe(false)
  })

  it('refuses a variable name that is not a lower-case custom property, and a preset without vars', () => {
    expect(parseLookPreset({ vars: { color: '#fff' } }).ok).toBe(false)
    expect(parseLookPreset({ vars: { '--Hello': '#fff' } }).ok).toBe(false)
    expect(parseLookPreset({}).ok).toBe(false)
    expect(parseLookPreset(null).ok).toBe(false)
  })

  it('keeps the accepted variables, frozen', () => {
    const parsed = parseLookPreset({ vars: { '--hello-accent': '#2f6fed', '--hello-gap': '12px' } })
    expect(parsed.ok).toBe(true)
    if (!parsed.ok) return
    expect(parsed.preset.vars).toEqual({ '--hello-accent': '#2f6fed', '--hello-gap': '12px' })
    expect(Object.isFrozen(parsed.preset.vars)).toBe(true)
  })
})
