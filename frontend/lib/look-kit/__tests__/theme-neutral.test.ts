// @vitest-environment node
/**
 * PRD-261 A-13a — the look runtime names no theme, and its move left nothing behind.
 *
 * - lib/look-kit's runtime files (not their tests) hold no theme's code or names: no import of
 *   Play's folders or of components, no --pl- variable, no 'play' interface, no automatos-play,
 *   no LOOK_INTERFACE. A theme hands the runtime its catalogue, prefix and spec (PA-9).
 * - lib/looks keeps none of the moved modules (css, contrast, hsl, validate, studio-tokens), and
 *   no prepaint-apply.ts exists in either folder (the plan's S2-19: it never existed).
 * - No source file in the frontend imports a moved module from its old place, by alias or by a
 *   relative path.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

const FRONTEND = path.resolve(__dirname, '../../..')
const KIT_DIR = path.join(FRONTEND, 'lib', 'look-kit')
const OLD_DIR = path.join(FRONTEND, 'lib', 'looks')
const MOVED: readonly string[] = ['css', 'contrast', 'hsl', 'validate', 'studio-tokens']
const NEVER_CREATED = 'prepaint-apply.ts'
const SOURCE_DIRS: readonly string[] = ['app', 'components', 'lib', 'hooks', 'contexts', 'stores', 'e2e']
const SKIP_DIRS = new Set(['node_modules'])
const TEST_DIR = '__tests__'
const SOURCE_FILE = /\.(ts|tsx)$/
/** ` from '…'`, `import('…')`, `vi.mock('…')` and a bare `import '…'`, in single or double quotes. */
const IMPORT_SPECIFIER = /(?:\sfrom\s+|\bimport\s*\(\s*|\bvi\.(?:do)?[mM]ock\(\s*|^\s*import\s+)(['"])([@.\w][^'"\s]*)\1/gm
const OLD_ALIAS = new RegExp(`^@/lib/looks/(${MOVED.join('|')})$`)

/** What a theme's own code or names look like; none may appear in the runtime. */
const THEME_MARKERS: readonly RegExp[] = [
  /--pl-/,
  /['"`]play['"`]/,
  /automatos-play/,
  /LOOK_INTERFACE/,
  /@\/lib\/(looks|play|mascot)(\/|['"])/,
  /@\/components\//,
]

/** Parsing every source file takes a few seconds; the default 5 s is too tight on a busy runner. */
const SCAN_TIMEOUT_MS = 120_000

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

function markersIn(source: string): string[] {
  return THEME_MARKERS.filter((marker) => marker.test(source)).map(String)
}

function specifiersOf(source: string): string[] {
  return Array.from(source.matchAll(IMPORT_SPECIFIER), (match) => match[2])
}

/** True when `specifier`, written in `file`, names a moved module at its old place. */
function reachesOldPlace(file: string, specifier: string): boolean {
  if (OLD_ALIAS.test(specifier)) return true
  if (!specifier.startsWith('.')) return false
  const target = path.resolve(path.dirname(file), specifier)
  return MOVED.some((name) => target === path.join(OLD_DIR, name) || target === path.join(OLD_DIR, `${name}.ts`))
}

const RUNTIME_FILES = walk(KIT_DIR, (file) => SOURCE_FILE.test(file) && !file.split(path.sep).includes(TEST_DIR))

describe('PRD-261 A-13a — the look runtime is theme-neutral', () => {
  it('has the moved modules', () => {
    const names = RUNTIME_FILES.map((file) => path.basename(file, '.ts'))
    expect(MOVED.filter((name) => !names.includes(name))).toEqual([])
  })

  it("holds no theme's code or names in its runtime files", () => {
    const found = RUNTIME_FILES.flatMap((file) => markersIn(readFileSync(file, 'utf8')).map((m) => `${relative(file)}: ${m}`))
    expect(found).toEqual([])
  })

  it("the markers catch a theme's code, so the scan can't pass by matching nothing", () => {
    const samples = [
      "import { GROUNDS } from '@/lib/looks/catalogue'",
      "const v = '--pl-accent'",
      "const i = 'play'",
      "import { LOOK_INTERFACE } from './types'",
      "import x from '@/components/play/frame'",
    ]
    expect(samples.filter((sample) => markersIn(sample).length === 0)).toEqual([])
  })
})

describe('PRD-261 A-13a — the move left nothing behind', () => {
  it('lib/looks keeps none of the moved modules, and no prepaint-apply.ts exists', () => {
    const left = MOVED.map((name) => path.join(OLD_DIR, `${name}.ts`)).filter((file) => existsSync(file))
    expect(left.map(relative)).toEqual([])
    expect(existsSync(path.join(OLD_DIR, NEVER_CREATED))).toBe(false)
    expect(existsSync(path.join(KIT_DIR, NEVER_CREATED))).toBe(false)
  })

  it('no source file imports a moved module from its old place', () => {
    const files = SOURCE_DIRS.flatMap((dir) => walk(path.join(FRONTEND, dir), (file) => SOURCE_FILE.test(file)))
    const stale = files.flatMap((file) => specifiersOf(readFileSync(file, 'utf8'))
      .filter((specifier) => reachesOldPlace(file, specifier))
      .map((specifier) => `${relative(file)}: ${specifier}`))
    expect(stale).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it('the import scan sees aliased and relative specifiers, and resolves the relative ones', () => {
    const file = path.join(OLD_DIR, 'style.ts')
    // Built by parts, so this file's own text never reads as an import of the old place to the scan.
    const oldCss = ['@/lib/looks', 'css'].join('/')
    expect(specifiersOf(`import { a } from '${oldCss}'\nvi.mock('./validate')`)).toEqual([oldCss, './validate'])
    expect(reachesOldPlace(file, oldCss)).toBe(true)
    expect(reachesOldPlace(file, './validate')).toBe(true)
    expect(reachesOldPlace(file, './engine')).toBe(false)
    expect(reachesOldPlace(path.join(KIT_DIR, 'css.ts'), './contrast')).toBe(false)
  })
})
