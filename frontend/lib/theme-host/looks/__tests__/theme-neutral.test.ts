// @vitest-environment node
/**
 * PRD-261 A-19d1, A-19d2 — the generic look runtime (lib/theme-host/looks) names no theme: its runtime files
 * (not their tests) pass the look runtime's THEME_MARKERS, the markers lib/look-kit/__tests__/
 * theme-neutral.test.ts holds lib/look-kit to (no theme's variable prefix, interface name, storage
 * keys or look format, no import of a theme's folders or of components). Both scans import the one
 * list, lib/look-kit/__tests__/theme-markers.ts (A-19f2; this scan used to parse it out of the
 * look-kit scan's source), so they can never drift; the scan must find them all, see every runtime
 * file and catch a theme's code, so it can't pass by matching nothing. The samples are built by
 * parts, so this file's own text never reads as a theme's code to the scan of the whole frontend
 * (lib/__tests__/prd261-no-theme-in-core.test.ts).
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { markersIn, THEME_MARKERS } from '@/lib/look-kit/__tests__/theme-markers'

const FRONTEND = path.resolve(__dirname, '../../../..')
const RUNTIME_DIR = path.join(FRONTEND, 'lib', 'theme-host', 'looks')
const TEST_DIR = '__tests__'
const SOURCE_FILE = /\.(ts|tsx)$/
/** The runtime's files A-19d1 and A-19d2 made: the scan must see each one. */
const MUST_SCAN: readonly string[] = [
  'spec', 'storage', 'state', 'store', 'server', 'transfer',
  'paint', 'tone-cookie', 'motion', 'page-boot', 'provider', 'bridge',
]
/** As many markers as the shared list holds today; it may grow, never shrink unnoticed. */
const MARKERS_AT_LEAST = 6

function walk(dir: string): string[] {
  if (!existsSync(dir)) return []
  return readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry)
    if (statSync(full).isDirectory()) return entry === TEST_DIR ? [] : walk(full)
    return SOURCE_FILE.test(full) ? [full] : []
  })
}

const RUNTIME_FILES = walk(RUNTIME_DIR)

describe('PRD-261 A-19d1 — the generic look runtime is theme-neutral', () => {
  it("shares every THEME_MARKERS entry of the look-kit scan", () => {
    expect(THEME_MARKERS.length).toBeGreaterThanOrEqual(MARKERS_AT_LEAST)
  })

  it('scans every runtime file and none of the tests', () => {
    const names = RUNTIME_FILES.map((file) => path.basename(file).replace(SOURCE_FILE, ''))
    expect(MUST_SCAN.filter((name) => !names.includes(name))).toEqual([])
    expect(RUNTIME_FILES.filter((file) => file.split(path.sep).includes(TEST_DIR))).toEqual([])
  })

  it("holds no theme's code or names in its runtime files", () => {
    const found = RUNTIME_FILES.flatMap((file) => markersIn(readFileSync(file, 'utf8')).map((marker) => `${path.relative(FRONTEND, file)}: ${marker}`))
    expect(found).toEqual([])
  })

  it("the markers catch a theme's code, so the scan can't pass by matching nothing", () => {
    const quoted = (text: string) => `'${text}'`
    const samples = [
      `import { GROUNDS } from ${quoted(['@', 'lib', 'looks', 'catalogue'].join('/'))}`,
      `const v = ${quoted(['-', '-pl-accent'].join(''))}`,
      `const i = ${quoted(['pl', 'ay'].join(''))}`,
      `const k = ${quoted(['automatos', 'pl' + 'ay.looks'].join('-'))}`,
      `import { ${['LOOK', 'INTERFACE'].join('_')} } from './types'`,
      `import x from ${quoted(['@', 'components', 'theme-host', 'look-scope'].join('/'))}`,
    ]
    expect(samples.filter((sample) => markersIn(sample).length === 0)).toEqual([])
  })
})
