// @vitest-environment node
/**
 * PRD-261 A-14d (W-A5) — Play's look editor reaches the core only through '@automatos/theme-sdk'.
 *
 * The area (plan §2.4: components/looks/** and the Play-specific
 * lib/looks/{catalogue,presets,document,editor,scope,style,transfer,types,tone}.ts; runtime files,
 * not their tests) imports 'react', 'react-dom', the SDK, and by relative path only Play's own
 * folders (its components, its lib files, the mascot library) and the look runtime the kit vendors
 * (lib/look-kit, and lib/looks/engine.ts that binds it to Play's catalogue, PA-9). Never '@/…',
 * 'next/*', '@tanstack/react-query', 'sonner', 'api-client' or a zustand store (PRD §4.3 item 7),
 * so W-K2 can copy the area into the Design Kit as part of the Play package.
 *
 * The host's own code that used to sit in components/looks left it in A-14d: the look runtime
 * (lib/theme-host/builtin/play-look-{provider,store,bridge}.tsx, components/theme-host/
 * account-look-scope.tsx), the "part of Play" note (lib/theme-host/builtin/play-only-route.tsx) and
 * the two lazy slots (lib/theme-host/builtin/play-{looks,style}-lazy.tsx). What Play copied from the
 * core instead is held to the core's own in components/looks/__tests__/copied-helpers.test.tsx.
 *
 * The specifiers are read through the TypeScript compiler (./module-references), so a comment or a
 * string naming a path never counts, and a type-only import does.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { moduleSpecifiers } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
const LOOKS_DIR = 'lib/looks'
const LOOK_FILES: readonly string[] = ['catalogue', 'presets', 'document', 'editor', 'scope', 'style', 'transfer', 'types', 'tone']
/** The look editor's components, whole: every runtime file under it is in the area. */
const EDITOR_DIR = 'components/looks'
const SOURCE_FILE = /\.(ts|tsx)$/
const TEST_FILE = /\.test\.tsx?$/
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
const ALLOWED_PACKAGES: ReadonlySet<string> = new Set(['@automatos/theme-sdk', 'react', 'react-dom'])

/** Where a relative import may land: Play's own folders, the mascot library, and the look runtime (PA-9). */
const ALLOWED_DIRS: readonly string[] = [EDITOR_DIR, 'components/play', 'components/mascot', LOOKS_DIR, 'lib/look-kit', 'lib/mascot']

/** Files the scan must find, so a moved folder can never pass by scanning nothing. */
const MUST_FIND: readonly string[] = [
  `${EDITOR_DIR}/looks-screen.tsx`,
  `${EDITOR_DIR}/look-manager.tsx`,
  `${EDITOR_DIR}/use-play-look.ts`,
  `${EDITOR_DIR}/style/style-screen.tsx`,
  `${LOOKS_DIR}/tone.ts`,
]
/** Imports the scan must see, so a broken reader can never pass by finding nothing. */
const MUST_SEE: readonly string[] = ['@automatos/theme-sdk', 'react', './use-play-look']

function runtimeFilesUnder(dir: string): string[] {
  return readdirSync(path.join(FRONTEND, dir)).flatMap((entry) => {
    const rel = `${dir}/${entry}`
    if (statSync(path.join(FRONTEND, rel)).isDirectory()) return entry === TEST_DIR ? [] : runtimeFilesUnder(rel)
    return SOURCE_FILE.test(entry) && !TEST_FILE.test(entry) ? [rel] : []
  })
}

/** The area's paths, relative to frontend/ (plan §2.4, W-A5 A-14d). */
const AREA_FILES: readonly string[] = [...runtimeFilesUnder(EDITOR_DIR), ...LOOK_FILES.map((name) => `${LOOKS_DIR}/${name}.ts`)]

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

function specifiersIn(file: string): string[] {
  return moduleSpecifiers(file, readFileSync(path.join(FRONTEND, file), 'utf8'))
}

function refusalsIn(file: string): string[] {
  return specifiersIn(file).flatMap((specifier) => {
    const why = refusal(file, specifier)
    return why === null ? [] : [`${file}: '${specifier}' (${why})`]
  })
}

describe('PRD-261 A-14d — the look editor imports the core only through the SDK', () => {
  it('scans the whole area: every listed file exists, and the editor folder is walked', () => {
    expect(AREA_FILES.filter((file) => !existsSync(path.join(FRONTEND, file)))).toEqual([])
    expect(MUST_FIND.filter((file) => !AREA_FILES.includes(file))).toEqual([])
    const seen = new Set(AREA_FILES.flatMap(specifiersIn))
    expect(MUST_SEE.filter((specifier) => !seen.has(specifier))).toEqual([])
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
      '../../components/theme-host/account-look-scope',
      'next-themes',
    ]
    expect(refused.filter((specifier) => refusal(file, specifier) === null)).toEqual([])
    const editor = `${EDITOR_DIR}/looks-screen.tsx`
    const outOfPlay = ['../../lib/theme-host/builtin/play-look-provider', '../../hooks/use-notice-seq', '../ui/button']
    expect(outOfPlay.filter((specifier) => refusal(editor, specifier) === null)).toEqual([])
  })

  it('the scan allows the SDK, React, and Play, look-kit and mascot paths', () => {
    const file = `${LOOKS_DIR}/style.ts`
    const allowed = ['@automatos/theme-sdk', 'react', 'react-dom', './catalogue', '../look-kit/contrast', '../mascot/ring']
    expect(allowed.map((specifier) => refusal(file, specifier))).toEqual(allowed.map(() => null))
    const editor = `${EDITOR_DIR}/style/style-components.tsx`
    const fromEditor = ['../../play/chat/labels', '../../mascot/auto-mascot', '../../../lib/looks/style', '../use-play-look']
    expect(fromEditor.map((specifier) => refusal(editor, specifier))).toEqual(fromEditor.map(() => null))
  })

  it('reads specifiers through the compiler, so comments and strings never count', () => {
    const source = "// import x from '@/lib/cookies'\nconst s = \"from 'sonner'\"\nimport { a } from './types'\n"
    expect(moduleSpecifiers('sample.ts', source)).toEqual(['./types'])
  })
})
