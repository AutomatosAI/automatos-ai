// @vitest-environment node
/**
 * PRD-261 A-19d1 — the generic look runtime (lib/theme-host/looks) names no theme: its runtime files
 * (not their tests) pass the look runtime's THEME_MARKERS, the markers lib/look-kit/__tests__/
 * theme-neutral.test.ts holds lib/look-kit to (no theme's variable prefix, interface name, storage
 * keys or look format, no import of a theme's folders or of components). The markers are read from
 * that file's source, so the two scans can never drift; the scan must find them all, see every
 * runtime file and catch a theme's code, so it can't pass by matching nothing. The samples are built
 * by parts, so this file's own text never reads as a theme's code to a scan of the whole frontend.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

const FRONTEND = path.resolve(__dirname, '../../../..')
const RUNTIME_DIR = path.join(FRONTEND, 'lib', 'theme-host', 'looks')
const MARKER_SOURCE = path.join(FRONTEND, 'lib', 'look-kit', '__tests__', 'theme-neutral.test.ts')
const TEST_DIR = '__tests__'
const SOURCE_FILE = /\.(ts|tsx)$/
/** The runtime's files A-19d1 made: the scan must see each one. */
const MUST_SCAN: readonly string[] = ['spec', 'storage', 'state', 'store', 'server', 'transfer']
/** The marker list's opening line in the look-kit scan, and one marker per line after it until `]`. */
const MARKERS_START = 'const THEME_MARKERS: readonly RegExp[] = ['
const MARKERS_END = ']'
const MARKER_LINE = /^\/(.+)\/([a-z]*),?$/
/** As many markers as the look-kit scan holds today; it may grow, never shrink unnoticed. */
const MARKERS_AT_LEAST = 6

function walk(dir: string): string[] {
  if (!existsSync(dir)) return []
  return readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry)
    if (statSync(full).isDirectory()) return entry === TEST_DIR ? [] : walk(full)
    return SOURCE_FILE.test(full) ? [full] : []
  })
}

/** The look-kit scan's THEME_MARKERS, parsed from its source line by line. */
function themeMarkers(): RegExp[] {
  const lines = readFileSync(MARKER_SOURCE, 'utf8').split('\n').map((line) => line.trim())
  const start = lines.indexOf(MARKERS_START)
  if (start < 0) return []
  const end = lines.indexOf(MARKERS_END, start)
  return lines.slice(start + 1, end < 0 ? start + 1 : end).flatMap((line) => {
    const match = MARKER_LINE.exec(line)
    return match === null ? [] : [new RegExp(match[1], match[2])]
  })
}

const MARKERS = themeMarkers()
const RUNTIME_FILES = walk(RUNTIME_DIR)

function markersIn(source: string): string[] {
  return MARKERS.filter((marker) => marker.test(source)).map(String)
}

describe('PRD-261 A-19d1 — the generic look runtime is theme-neutral', () => {
  it("reads every THEME_MARKERS line of the look-kit scan", () => {
    expect(MARKERS.length).toBeGreaterThanOrEqual(MARKERS_AT_LEAST)
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
