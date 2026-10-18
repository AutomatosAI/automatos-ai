/**
 * PRD-261 A-01, A-04a — an installed theme's description and look preset are checked before
 * anything is applied: the active-theme answer must name the install the style asks for, with a
 * sha256 digest; the core's own <html> classes are refused; every file must resolve under that
 * digest's files folder on this origin; and a preset value is a colour, a length or a number.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { parseActiveInstall, parseLookPreset, parseSpikeManifest, resolveUnderRoot, themeFilesRoot } from '../manifest'
import { HTML_CLASS_OWNERS, RESERVED_HTML_CLASSES } from '../package-manifest'
import { STYLE_HTML_CLASS_NAMES } from '@/lib/ui-style'

/** The hello theme's manifest, as the browser lane serves it. */
const FIXTURES = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-theme')
const HELLO_MANIFEST = JSON.parse(readFileSync(path.join(FIXTURES, 'theme.json'), 'utf8')) as Readonly<Record<string, unknown>>
const MISMATCHED_MANIFEST: unknown = JSON.parse(readFileSync(path.join(FIXTURES, 'theme.engines-mismatch.json'), 'utf8'))
const DIGEST = '0123456789abcdef'.repeat(4)
const OTHER_DIGEST = 'fedcba9876543210'.repeat(4)
const ROOT = `/api/themes/files/${DIGEST}/`
const INSTALL_ID = '00000000-0000-4000-8000-00000000e2e1'

function withField(field: string, value: unknown): Record<string, unknown> {
  return { ...HELLO_MANIFEST, [field]: value }
}

function detailOf(raw: unknown): string | null {
  const parsed = parseSpikeManifest(raw, ROOT)
  return parsed.ok ? null : parsed.detail
}

describe('PRD-261 A-04a — parseSpikeManifest', () => {
  it("reads the hello theme's manifest, with its files resolved under the digest's folder", () => {
    const parsed = parseSpikeManifest(HELLO_MANIFEST, ROOT)
    expect(parsed.ok).toBe(true)
    if (!parsed.ok) return
    const root = `${window.location.origin}${ROOT}`
    expect(parsed.manifest).toMatchObject({ id: 'automatos-e2e/hello', htmlClass: 'hello-spike', sdkRange: '^1.0.0', slots: ['frame@1'] })
    expect(parsed.manifest.entry).toBe(`${root}entry.js`)
    expect(parsed.manifest.styleSheet).toBe(`${root}theme.css`)
    expect(parsed.manifest.preset).toBe(`${root}presets/default.look.json`)
  })

  it('reads the engines-mismatch manifest too: the range is checked by the loader, not here', () => {
    const parsed = parseSpikeManifest(MISMATCHED_MANIFEST, ROOT)
    expect(parsed.ok && parsed.manifest.sdkRange).toBe('^2.0.0')
  })

  it('reserves every Style-axis class the core sets on <html>', () => {
    for (const name of STYLE_HTML_CLASS_NAMES) expect(RESERVED_HTML_CLASSES, name).toContain(name)
    expect(RESERVED_HTML_CLASSES).toEqual(expect.arrayContaining(['classic', 'dark', 'light', 'system']))
  })

  it("refuses every class the core owns on <html> as the theme's class", () => {
    for (const reserved of RESERVED_HTML_CLASSES) {
      expect(detailOf(withField('htmlClass', reserved)), reserved).not.toBeNull()
    }
  })

  it('PA-8: lets only the owning package keep its reserved class', () => {
    for (const [htmlClass, ownerId] of HTML_CLASS_OWNERS) {
      expect(detailOf({ ...HELLO_MANIFEST, id: ownerId, htmlClass }), htmlClass).toBeNull()
      expect(detailOf({ ...HELLO_MANIFEST, id: `${ownerId}-fork`, htmlClass }), htmlClass).not.toBeNull()
    }
  })

  it.each(['Hello', 'hi', 'hello spike', 'hello_spike', '1hello', 'a'.repeat(33), 42])('refuses the htmlClass %j', (htmlClass) => {
    expect(detailOf(withField('htmlClass', htmlClass))).not.toBeNull()
  })

  it.each([
    '../entry.js',
    '/entry.js',
    '//evil.example/x.js',
    'https://evil.example/x.js',
    'entry.js?x=:',
    'a/../../b.js',
    'a\\b.js',
    `../${OTHER_DIGEST}/entry.js`,
    `/api/themes/files/${OTHER_DIGEST}/entry.js`,
    '',
    7,
  ])(
    'refuses the entry %j',
    (entry) => {
      expect(detailOf(withField('entry', entry))).not.toBeNull()
    },
  )

  it('accepts every slot the host serves, by name', () => {
    expect(detailOf(withField('slots', ['frame@1', 'lookEditor@1', 'styleSheet@1', 'notFound@1']))).toBeNull()
  })

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

describe('PRD-261 A-04a — resolveUnderRoot and themeFilesRoot', () => {
  it("resolves a plain relative path under the given folder on this origin", () => {
    expect(resolveUnderRoot('presets/default.look.json', ROOT)).toBe(`${window.location.origin}${ROOT}presets/default.look.json`)
    expect(resolveUnderRoot('C:/x.js', ROOT)).toBeNull()
    expect(resolveUnderRoot('entry.js#x', ROOT)).toBeNull()
  })

  it('gives the files folder of a sha256 hex digest, and nothing for any other value', () => {
    expect(themeFilesRoot(DIGEST)).toBe(ROOT)
    for (const bad of [DIGEST.toUpperCase(), DIGEST.slice(1), `${DIGEST}0`, `../${DIGEST.slice(3)}`, '', null, 42]) {
      expect(themeFilesRoot(bad), String(bad)).toBeNull()
    }
  })
})

describe('PRD-261 A-04a — parseActiveInstall', () => {
  const active = (theme: unknown) => ({ choice: INSTALL_ID, theme })
  const theme = { installId: INSTALL_ID, digest: DIGEST, manifest: HELLO_MANIFEST }

  it("takes the install's digest, files folder and manifest when it is the install the style names", () => {
    const parsed = parseActiveInstall(active(theme), INSTALL_ID)
    expect(parsed).toEqual({ ok: true, digest: DIGEST, filesRoot: ROOT, manifest: HELLO_MANIFEST })
  })

  it('refuses no theme, another install, a bad digest and an answer that is not an object', () => {
    expect(parseActiveInstall(active(null), INSTALL_ID).ok).toBe(false)
    expect(parseActiveInstall(active({ ...theme, installId: '00000000-0000-4000-8000-000000000000' }), INSTALL_ID).ok).toBe(false)
    expect(parseActiveInstall(active({ ...theme, digest: '../etc' }), INSTALL_ID).ok).toBe(false)
    expect(parseActiveInstall(active({ ...theme, digest: undefined }), INSTALL_ID).ok).toBe(false)
    expect(parseActiveInstall(null, INSTALL_ID).ok).toBe(false)
    expect(parseActiveInstall([theme], INSTALL_ID).ok).toBe(false)
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
