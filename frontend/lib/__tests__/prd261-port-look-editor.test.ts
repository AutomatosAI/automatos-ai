// @vitest-environment node
/**
 * PRD-261 A-14d (W-A5) — Play's look editor reaches the core only through '@automatos/theme-sdk'.
 *
 * The area's files import 'react', 'react-dom', the SDK, and by relative path only Play's own looks
 * files, the look runtime (lib/look-kit, PA-9: the kit vendors it) and the mascot library. Never '@/…',
 * 'next/*', '@tanstack/react-query', 'sonner', 'api-client' or a zustand store (PRD §4.3 item 7), so
 * W-K2 can copy the area into the Design Kit as part of the Play package.
 *
 * The specifiers are read through the TypeScript compiler (./module-references), so a comment or a
 * string naming a path never counts. Part 1 (A-14d1) covers the Play-specific lib/looks files;
 * components/looks/** joins the area in A-14d2.
 */
import { existsSync, readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { moduleSpecifiers } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
const LOOKS_DIR = 'lib/looks'
const LOOK_FILES: readonly string[] = ['catalogue', 'presets', 'document', 'editor', 'scope', 'style', 'transfer', 'types', 'tone']

/** The area's paths, relative to frontend/ (plan §2.4, W-A5 A-14d). */
const AREA_FILES: readonly string[] = LOOK_FILES.map((name) => `${LOOKS_DIR}/${name}.ts`)

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
const ALLOWED_PACKAGES: ReadonlySet<string> = new Set(['@automatos/theme-sdk', 'react', 'react-dom'])

/** Where a relative import may land: Play's looks files, the look runtime (PA-9) and the mascot library. */
const ALLOWED_DIRS: readonly string[] = [LOOKS_DIR, 'lib/look-kit', 'lib/mascot']

function isInside(target: string, dir: string): boolean {
  const from = path.relative(path.join(FRONTEND, dir), target)
  return from === '' || (!from.startsWith('..') && !path.isAbsolute(from))
}

/** Why `specifier`, written in the area file `file` (relative to frontend/), is refused; null when it is allowed. */
function refusal(file: string, specifier: string): string | null {
  if (BANNED.some((pattern) => pattern.test(specifier))) return 'banned'
  if (specifier.startsWith('.')) {
    const target = path.resolve(FRONTEND, path.dirname(file), specifier)
    return ALLOWED_DIRS.some((dir) => isInside(target, dir)) ? null : 'outside Play and the look runtime'
  }
  return ALLOWED_PACKAGES.has(specifier) ? null : 'not an allowed package'
}

function refusalsIn(file: string): string[] {
  const source = readFileSync(path.join(FRONTEND, file), 'utf8')
  return moduleSpecifiers(file, source).flatMap((specifier) => {
    const why = refusal(file, specifier)
    return why === null ? [] : [`${file}: '${specifier}' (${why})`]
  })
}

describe('PRD-261 A-14d — the look editor imports the core only through the SDK', () => {
  it('every file of the area exists, so the scan cannot pass on a missing file', () => {
    expect(AREA_FILES.filter((file) => !existsSync(path.join(FRONTEND, file)))).toEqual([])
  })

  it('no file of the area imports a banned module or reaches past Play and the look runtime', () => {
    expect(AREA_FILES.flatMap(refusalsIn)).toEqual([])
  })

  it('the scan refuses every banned kind, by alias and by a relative path out of Play', () => {
    const file = `${LOOKS_DIR}/tone.ts`
    const refused = [
      '@/lib/cookies',
      'next/navigation',
      'next',
      '@tanstack/react-query',
      'sonner',
      'zustand',
      'zustand/middleware',
      '../api-client',
      '../cookies',
      '../theme-host/boot-sanitize',
      '../../components/ui/button',
      'next-themes',
    ]
    expect(refused.filter((specifier) => refusal(file, specifier) === null)).toEqual([])
  })

  it('the scan allows the SDK, React, and Play, look-kit and mascot paths', () => {
    const file = `${LOOKS_DIR}/style.ts`
    const allowed = ['@automatos/theme-sdk', 'react', 'react-dom', './catalogue', '../look-kit/contrast', '../mascot/ring']
    expect(allowed.map((specifier) => refusal(file, specifier))).toEqual(allowed.map(() => null))
  })

  it('reads specifiers through the compiler, so comments and strings never count', () => {
    const source = "// import x from '@/lib/cookies'\nconst s = \"from 'sonner'\"\nimport { a } from './types'\n"
    expect(moduleSpecifiers('sample.ts', source)).toEqual(['./types'])
  })
})
