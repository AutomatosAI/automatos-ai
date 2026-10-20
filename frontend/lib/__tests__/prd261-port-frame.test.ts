// @vitest-environment node
/**
 * PRD-261 A-14f (W-A5) — Play's frame and edges reach the core only through '@automatos/theme-sdk'.
 *
 * The area (components/play/shell/**, components/play/edges/**, plan §2.4) imports 'react',
 * 'react-dom', the SDK, lucide-react (Play's icons: a library, not the core) and, by relative path,
 * only Play's own files (components/play, components/mascot, Play's hooks, lib/play) and the mascot
 * library (lib/mascot). Never '@/…', 'next/*', '@tanstack/react-query', 'sonner', 'api-client' or a
 * zustand store (PRD §4.3 item 7), so W-K2 can copy the area into the Design Kit with the Play package.
 *
 * Two files are the core's own seam into builtin Play, not Play's code: the lazy slots
 * (play-frame-lazy.tsx, lost-screen-lazy.tsx) that lib/theme-host/builtin/play-slots.tsx registers,
 * each a next/dynamic import() of one Play screen (PRD-260 M17; prd260-play-guards.test.ts holds
 * them where they are). They load nothing else; an installed Play registers its screens with
 * defineTheme instead (A-16b), so they never leave the core.
 *
 * The specifiers are read through the TypeScript compiler (./module-references), so a comment or a
 * string naming a path never counts. Tests are not part of the scan: they drive the core around Play.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { PLAY_CONVERSATIONS_HREF as PLAY_OWN_CONVERSATIONS_HREF } from '@/components/play/shell/play-routes'
import { PLAY_CONVERSATIONS_HREF } from '@/lib/theme-host/builtin/play-entry'

import { moduleSpecifiers } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')

/** The area's folders, relative to frontend/ (plan §2.4, W-A5 A-14f). */
const AREA_DIRS: readonly string[] = ['components/play/shell', 'components/play/edges']
/** Play's own hook only the asleep screen uses (PRD-260 §4): no lane lists it, so it is scanned with the area. */
const AREA_EXTRA_FILES: readonly string[] = ['hooks/use-recheck-countdown.ts']
/** The core's lazy slots inside the area (see the header). */
const LAZY_SLOT = /-lazy\.tsx$/
const LAZY_LOADER = 'next/dynamic'

const SOURCE_FILE = /\.tsx?$/
const TEST_DIR = '__tests__'

/** What the area may never import: the core by alias, Next, the core's data and toast libraries, its stores. */
const BANNED: readonly RegExp[] = [
  /^@\//,
  /^next(\/|$)/,
  /^@tanstack\/react-query(\/|$)/,
  /^sonner(\/|$)/,
  /^zustand(\/|$)/,
  /api-client/,
]

/** The packages the area may import by name. */
const ALLOWED_PACKAGES: ReadonlySet<string> = new Set(['@automatos/theme-sdk', 'react', 'react-dom', 'lucide-react'])

/** Where a relative import may land: Play's own folders and files, and the mascot library. */
const ALLOWED_DIRS: readonly string[] = ['components/play', 'components/mascot', 'lib/play', 'lib/mascot']
const ALLOWED_FILES: readonly string[] = ['hooks/use-backend-mascot.ts', ...AREA_EXTRA_FILES]
const MODULE_SUFFIXES: readonly string[] = ['', '.ts', '.tsx']

function walk(dir: string): string[] {
  return readdirSync(path.join(FRONTEND, dir)).flatMap((name) => {
    const rel = `${dir}/${name}`
    if (statSync(path.join(FRONTEND, rel)).isDirectory()) return name === TEST_DIR ? [] : walk(rel)
    return SOURCE_FILE.test(name) ? [rel] : []
  })
}

/** Every source file of the area, relative to frontend/, tests left out. */
function areaFiles(): string[] {
  return [...AREA_DIRS.flatMap(walk), ...AREA_EXTRA_FILES]
}

function isInside(target: string, dir: string): boolean {
  const from = path.relative(path.join(FRONTEND, dir), target)
  return from === '' || (!from.startsWith('..') && !path.isAbsolute(from))
}

function isAllowedFile(target: string): boolean {
  return ALLOWED_FILES.some((file) => MODULE_SUFFIXES.some((suffix) => target + suffix === path.join(FRONTEND, file)))
}

/** Why `specifier`, written in the area file `file` (relative to frontend/), is refused; null when it is allowed. */
function refusal(file: string, specifier: string): string | null {
  if (BANNED.some((pattern) => pattern.test(specifier))) return 'banned'
  if (specifier.startsWith('.')) {
    const target = path.resolve(FRONTEND, path.dirname(file), specifier)
    return ALLOWED_DIRS.some((dir) => isInside(target, dir)) || isAllowedFile(target) ? null : 'outside Play'
  }
  return ALLOWED_PACKAGES.has(specifier) ? null : 'not an allowed package'
}

function specifiersOf(file: string): string[] {
  return moduleSpecifiers(file, readFileSync(path.join(FRONTEND, file), 'utf8'))
}

function refusalsIn(file: string): string[] {
  return specifiersOf(file).flatMap((specifier) => {
    const why = refusal(file, specifier)
    return why === null ? [] : [`${file}: '${specifier}' (${why})`]
  })
}

/** A lazy slot loads next/dynamic and one module of its own folder, nothing else. */
function slotProblems(file: string): string[] {
  const specifiers = specifiersOf(file)
  const own = specifiers.filter((specifier) => specifier.startsWith('./') && !specifier.slice(2).includes('/'))
  const other = specifiers.filter((specifier) => specifier !== LAZY_LOADER && !own.includes(specifier))
  return [...other.map((specifier) => `${file}: '${specifier}'`), ...(own.length === 1 ? [] : [`${file}: ${own.length} own modules`])]
}

describe('PRD-261 A-14f — the frame and edges import the core only through the SDK', () => {
  it('the area has its files, so the scan cannot pass on a moved folder', () => {
    expect(AREA_DIRS.filter((dir) => !existsSync(path.join(FRONTEND, dir)))).toEqual([])
    expect(AREA_EXTRA_FILES.filter((file) => !existsSync(path.join(FRONTEND, file)))).toEqual([])
    const files = areaFiles()
    expect(files).toEqual(expect.arrayContaining(['components/play/shell/play-frame.tsx', 'components/play/edges/asleep-gate.tsx']))
    expect(files.filter((file) => LAZY_SLOT.test(file)).sort()).toEqual([
      'components/play/edges/lost-screen-lazy.tsx',
      'components/play/shell/play-frame-lazy.tsx',
    ])
  })

  it('no file of the area imports a banned module or reaches past Play and the mascot library', () => {
    const files = areaFiles().filter((file) => !LAZY_SLOT.test(file))
    expect(files.flatMap(refusalsIn)).toEqual([])
  })

  it("the core's lazy slots load next/dynamic and one Play screen beside them, nothing else", () => {
    const slots = areaFiles().filter((file) => LAZY_SLOT.test(file))
    expect(slots.flatMap(slotProblems)).toEqual([])
  })

  it('the scan refuses every banned kind, by alias and by a relative path out of Play', () => {
    const file = 'components/play/shell/play-sidebar.tsx'
    const refused = [
      '@/hooks/use-backend-health',
      'next/link',
      'next/navigation',
      'next',
      '@tanstack/react-query',
      'sonner',
      'zustand',
      '../../../lib/api-client',
      '../../../stores/chat-session-store',
      '../../../hooks/use-start-chat',
      '../../layout/studio-header',
      '../../../lib/ticket-links',
      'next-themes',
    ]
    expect(refused.filter((specifier) => refusal(file, specifier) === null)).toEqual([])
  })

  it("the scan allows the SDK, React, Play's icons, and Play's and the mascot library's paths", () => {
    const file = 'components/play/edges/asleep-screen.tsx'
    const allowed = [
      '@automatos/theme-sdk',
      'react',
      'react-dom',
      'lucide-react',
      './edges.module.css',
      '../shell/page-columns',
      '../../mascot/auto-mascot',
      '../../../lib/play/focus',
      '../../../lib/mascot/types',
      '../../../hooks/use-backend-mascot',
      '../../../hooks/use-recheck-countdown',
    ]
    expect(allowed.map((specifier) => refusal(file, specifier))).toEqual(allowed.map(() => null))
  })

  it("Play's own Conversations address is the one builtin Play registers", () => {
    expect(PLAY_OWN_CONVERSATIONS_HREF).toBe(PLAY_CONVERSATIONS_HREF)
  })
})
