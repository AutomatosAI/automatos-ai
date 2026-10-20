// @vitest-environment node
/**
 * PRD-261 A-14e (W-A5) — Play's mascot reaches the core only through the SDK, so it can leave
 * core as part of the Play package (W-K2, A-16b).
 *
 * A static scan of exactly the area's paths (components/mascot/**, its tests included, and
 * hooks/use-{auto,backend}-mascot.ts, hooks/use-mascot-player.ts, hooks/use-media-preferences.ts),
 * read through the TypeScript compiler (./module-references), so a comment or a string never counts
 * and vi.mock() ids are read as well as imports:
 *
 * - no `@/…`, `next/*`, `@tanstack/react-query`, `sonner`, `zustand` or `api-client` specifier;
 * - a package import is `react`, `react-dom` or `@automatos/theme-sdk` (a test may add vitest and
 *   Testing Library);
 * - a relative import lands in Play's mascot code: components/mascot, the mascot library
 *   (lib/mascot) or one of Play's mascot hooks, the idle timer included (sdk-surface-map.md:
 *   `@/hooks/use-idle-timeout` is copied into Play with the driver that uses it).
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { moduleSpecifiers } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
const AREA_DIRS: readonly string[] = ['components/mascot']
const AREA_FILES: readonly string[] = [
  'hooks/use-auto-mascot.ts',
  'hooks/use-backend-mascot.ts',
  'hooks/use-mascot-player.ts',
  'hooks/use-media-preferences.ts',
]
/** Where a relative import of the area may land: Play's mascot code, which moves with Play. */
const PLAY_MASCOT_DIRS: readonly string[] = ['components/mascot/', 'lib/mascot/']
const PLAY_MASCOT_FILES: readonly string[] = [...AREA_FILES, 'hooks/use-idle-timeout.ts']

const BANNED: readonly RegExp[] = [
  /^@\//,
  /^next(\/|$)/,
  /^@tanstack\/react-query(\/|$)/,
  /^sonner(\/|$)/,
  /^zustand(\/|$)/,
  /api-client/,
]
const ALLOWED_PACKAGES: readonly RegExp[] = [/^react(\/|$)/, /^react-dom(\/|$)/, /^@automatos\/theme-sdk$/]
const ALLOWED_IN_TESTS: readonly RegExp[] = [/^vitest$/, /^@testing-library\//]
/** Imports the scan must see, so a broken reader can never pass by finding nothing. */
const KNOWN_IMPORTS: readonly string[] = ['react', '@automatos/theme-sdk', '../../lib/mascot/media', './use-idle-timeout']
const KNOWN_FILES: readonly string[] = ['components/mascot/auto-mascot.tsx', 'components/mascot/use-mascot-motion.ts', ...AREA_FILES]

const SOURCE_FILE = /\.(ts|tsx)$/
const TEST_FILE = /(\.test\.tsx?$)|(\/__tests__\/)/
const MODULE_SUFFIXES: readonly string[] = ['', '.ts', '.tsx', '/index.ts', '/index.tsx']

interface Found {
  readonly file: string
  readonly specifier: string
}

function toPosix(file: string): string {
  return file.split(path.sep).join('/')
}

function sourcesUnder(dir: string): string[] {
  return readdirSync(path.join(FRONTEND, dir)).flatMap((entry) => {
    const rel = `${dir}/${entry}`
    if (statSync(path.join(FRONTEND, rel)).isDirectory()) return sourcesUnder(rel)
    return SOURCE_FILE.test(entry) ? [rel] : []
  })
}

function areaFiles(): string[] {
  return [...AREA_DIRS.flatMap(sourcesUnder), ...AREA_FILES].sort()
}

function importsOf(files: readonly string[]): Found[] {
  return files.flatMap((file) => {
    const source = readFileSync(path.join(FRONTEND, file), 'utf8')
    return moduleSpecifiers(file, source).map((specifier) => ({ file, specifier }))
  })
}

function isPackage(specifier: string): boolean {
  return !specifier.startsWith('.')
}

function packageAllowed({ file, specifier }: Found): boolean {
  const allowed = TEST_FILE.test(file) ? [...ALLOWED_PACKAGES, ...ALLOWED_IN_TESTS] : ALLOWED_PACKAGES
  return allowed.some((pattern) => pattern.test(specifier))
}

/** The frontend file a relative specifier names, relative to the frontend; null when none exists. */
function resolveRelative({ file, specifier }: Found): string | null {
  const base = path.resolve(path.dirname(path.join(FRONTEND, file)), specifier)
  const hit = MODULE_SUFFIXES.map((suffix) => base + suffix).find(
    (candidate) => existsSync(candidate) && statSync(candidate).isFile(),
  )
  return hit === undefined ? null : toPosix(path.relative(FRONTEND, hit))
}

function landsInPlayMascot(found: Found): boolean {
  const target = resolveRelative(found)
  if (target === null) return false
  return PLAY_MASCOT_FILES.includes(target) || PLAY_MASCOT_DIRS.some((dir) => target.startsWith(dir))
}

const describeFound = ({ file, specifier }: Found): string => `${file}: '${specifier}'`

const FILES = areaFiles()
const FOUND = importsOf(FILES)

describe('PRD-261 A-14e — the scan of the mascot area', () => {
  it('reads imports, export-from, import(), import types and vi.mock() ids, never comments or strings', () => {
    const source = [
      "import { a } from '@/lib/mascot/media'",
      "export { b } from 'next/link'",
      "type C = import('sonner').Toaster",
      "const d = () => import('zustand')",
      "vi.mock('@tanstack/react-query', () => ({}))",
      "// import { e } from '@/lib/api-client'",
      "const words = \"from '@/hooks/x'\"",
    ].join('\n')
    expect(moduleSpecifiers('sample.tsx', source)).toEqual([
      '@/lib/mascot/media', 'next/link', 'sonner', 'zustand', '@tanstack/react-query',
    ])
  })

  it("covers the area's files and sees the imports it must see", () => {
    expect(KNOWN_FILES.filter((file) => !FILES.includes(file))).toEqual([])
    const specifiers = FOUND.map((found) => found.specifier)
    expect(KNOWN_IMPORTS.filter((specifier) => !specifiers.includes(specifier))).toEqual([])
  })
})

describe("PRD-261 A-14e — Play's mascot imports the core only through the SDK", () => {
  it('names no @/…, next/*, react-query, sonner, zustand or api-client module, in code or in a mock', () => {
    const banned = FOUND.filter(({ specifier }) => BANNED.some((pattern) => pattern.test(specifier)))
    expect(banned.map(describeFound)).toEqual([])
  })

  it('imports no package but react, react-dom and @automatos/theme-sdk (and the test runner in tests)', () => {
    const other = FOUND.filter((found) => isPackage(found.specifier) && !packageAllowed(found))
    expect(other.map(describeFound)).toEqual([])
  })

  it("reaches by a relative path only Play's mascot code: components/mascot, lib/mascot and its hooks", () => {
    const outside = FOUND.filter((found) => !isPackage(found.specifier) && !landsInPlayMascot(found))
    expect(outside.map(describeFound)).toEqual([])
  })
})
