/**
 * PRD-261 A-01 — the theme import map maps each shared specifier to its shim under the public
 * path, and its script body is the existing scriptSafeJson output, which holds no '<'.
 */
import { describe, expect, it } from 'vitest'

import { scriptSafeJson } from '@/lib/looks/prepaint'
import sharedModules from '../shared-modules.json'
import { IMPORT_MAP_SCRIPT_BODY, THEME_IMPORT_MAP } from '../import-map'

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
