// @vitest-environment node
/**
 * PRD-261 A-13a — the look runtime names no theme.
 *
 * - lib/look-kit's runtime files (not their tests) hold no theme's code or names: no import of a
 *   theme's folders or of components, no theme variable prefix, no quoted theme interface, no theme
 *   storage key, no theme look format (./theme-markers.ts, which the generic look runtime's scan
 *   shares). A theme hands the runtime its catalogue, prefix and spec (PA-9).
 * - The modules A-13a moved in (css, contrast, hsl, validate, studio-tokens) and the look scope A-19a
 *   moved in (scope) are all here, so the scan holds them; no prepaint-apply.ts exists (the plan's
 *   S2-19: it never existed). QB-19 (PB-38): core keeps every one of them after W-A8.
 *
 * PRD-261 A-19f2: builtin Play's lib/looks, where these modules came from, is gone with builtin Play
 * (A-19f1), so the old checks that it kept none of them and that no file imported one from there are
 * lib/__tests__/prd261-no-theme-in-core.test.ts's now, over the whole frontend: no file is left under
 * lib/looks and no file imports anything there.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { markersIn } from './theme-markers'

const FRONTEND = path.resolve(__dirname, '../../..')
const KIT_DIR = path.join(FRONTEND, 'lib', 'look-kit')
const MOVED: readonly string[] = ['css', 'contrast', 'hsl', 'validate', 'studio-tokens']
const NEVER_CREATED = 'prepaint-apply.ts'
/** Moved in by A-19a (git mv of the old look scope). */
const MOVED_IN_A19A: readonly string[] = ['scope']
const SKIP_DIRS = new Set(['node_modules'])
const TEST_DIR = '__tests__'
const SOURCE_FILE = /\.(ts|tsx)$/

function walk(dir: string, keep: (file: string) => boolean): string[] {
  if (!existsSync(dir)) return []
  return readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry)
    if (statSync(full).isDirectory()) return SKIP_DIRS.has(entry) ? [] : walk(full, keep)
    return keep(full) ? [full] : []
  })
}

function relative(file: string): string {
  return path.relative(FRONTEND, file).split(path.sep).join('/')
}

const RUNTIME_FILES = walk(KIT_DIR, (file) => SOURCE_FILE.test(file) && !file.split(path.sep).includes(TEST_DIR))
const RUNTIME_NAMES = RUNTIME_FILES.map((file) => path.basename(file, '.ts'))

describe('PRD-261 A-13a — the look runtime is theme-neutral', () => {
  it('has the moved modules', () => {
    expect(MOVED.filter((name) => !RUNTIME_NAMES.includes(name))).toEqual([])
  })

  it('scans the look scope A-19a moved in, so the markers hold it', () => {
    expect(MOVED_IN_A19A.filter((name) => !RUNTIME_NAMES.includes(name))).toEqual([])
  })

  it("holds no theme's code or names in its runtime files", () => {
    const found = RUNTIME_FILES.flatMap((file) => markersIn(readFileSync(file, 'utf8')).map((marker) => `${relative(file)}: ${marker}`))
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

describe('PRD-261 A-13a — no prepaint-apply.ts was ever made', () => {
  it('exists in neither the look runtime nor its old folder', () => {
    expect(existsSync(path.join(KIT_DIR, NEVER_CREATED))).toBe(false)
    expect(existsSync(path.join(FRONTEND, 'lib', 'looks', NEVER_CREATED))).toBe(false)
  })
})
