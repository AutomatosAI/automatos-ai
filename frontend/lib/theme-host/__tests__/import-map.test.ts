/**
 * PRD-261 A-01 — the theme import map maps each shared specifier to its shim under the public
 * path, and its script body is the existing scriptSafeJson output, which holds no '<'.
 * PRD-261 A-03 (PA-13) — its integrity block carries only checked entries.
 */
import { describe, expect, it } from 'vitest'

import { scriptSafeJson } from '@/lib/theme-host/script-safe-json'
import sharedModules from '../shared-modules.json'
import { IMPORT_MAP_SCRIPT_BODY, THEME_IMPORT_MAP, importMapScriptBody, sanitizeIntegrity } from '../import-map'

const DIGEST = 'a'.repeat(64)
const MODULE_URL = `/api/themes/files/${DIGEST}/dist/entry.js`
const SHA256 = `sha256-${'A'.repeat(43)}=`

describe('PRD-261 A-01 — the theme import map', () => {
  it('maps every shared specifier to its shim under the public path, and nothing else', () => {
    const expected = Object.fromEntries(sharedModules.modules.map((entry) => [entry.specifier, `${sharedModules.publicPath}${entry.file}`]))
    expect(THEME_IMPORT_MAP.imports).toEqual(expected)
    expect(Object.keys(THEME_IMPORT_MAP)).toEqual(['imports'])
    expect(Object.isFrozen(THEME_IMPORT_MAP)).toBe(true)
    expect(Object.isFrozen(THEME_IMPORT_MAP.imports)).toBe(true)
  })

  it('has a script body that parses back to the map and holds no "<"', () => {
    expect(JSON.parse(IMPORT_MAP_SCRIPT_BODY)).toEqual(THEME_IMPORT_MAP)
    expect(IMPORT_MAP_SCRIPT_BODY).not.toContain('<')
  })

  it('escapes "<" and the line separators the way the body relies on', () => {
    expect(scriptSafeJson('</script><!--\u2028')).toBe('"\\u003c/script>\\u003c!--\\u2028"')
  })
})

describe('PRD-261 A-03 — the import map on every page, with integrity when a theme brings it', () => {
  it('is the constant body when there is no integrity, or none of it passes', () => {
    for (const integrity of [null, undefined, {}, [], 'x', { [MODULE_URL]: 'sha256-short=' }]) {
      expect(importMapScriptBody(integrity)).toBe(IMPORT_MAP_SCRIPT_BODY)
    }
  })

  it('keeps only module URLs of the theme files route with a sha256 value', () => {
    const integrity = {
      [MODULE_URL]: SHA256,
      [`/api/themes/files/${DIGEST}/../../api/admin.js`]: SHA256,
      [`/api/themes/files/${DIGEST}/style.css`]: SHA256,
      'https://evil.test/x.js': SHA256,
      [`/api/themes/files/${DIGEST}/b.js`]: '</script>',
    }
    expect(sanitizeIntegrity(integrity)).toEqual({ [MODULE_URL]: SHA256 })
    const body = importMapScriptBody(integrity)
    expect(JSON.parse(body)).toEqual({ ...THEME_IMPORT_MAP, integrity: { [MODULE_URL]: SHA256 } })
    expect(body).not.toContain('<')
  })
})
