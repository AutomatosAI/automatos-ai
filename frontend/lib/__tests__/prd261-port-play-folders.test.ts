// @vitest-environment node
/**
 * PRD-261 W-A5 (before W-K2, A-16b) — Play's six folders, whole, import nothing by the core's alias.
 *
 * W-K2 copies components/{play,looks,mascot} and lib/{play,looks,mascot} into the Design Kit at W-A5's
 * merge SHA, and the package must have no `@/` import. The area guards (prd261-port-<area>.test.ts) each
 * hold their own paths; this one scans every runtime file of the six folders (their tests and
 * __tests__ folders left out), so a file no lane lists (lib/looks/engine.ts, lib/mascot/**) is held
 * too. Each specifier is read through the TypeScript compiler (./module-references): imports,
 * export-from, import(), import types and vi.mock() ids, never a comment or a string. The ban list is
 * the area guards' own: `@/…`, `next/*`, `@tanstack/react-query`, `sonner`, `zustand`, `api-client`.
 *
 * Host code that sat in Play's folders left them in W-A5 (git mv, no copy): builtin Play's prepaint
 * table builder, the look storage and the look state are in lib/theme-host/builtin/.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { moduleSpecifiers } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
/** The folders W-K2 copies into the Play package (plan §2.4, A-16b). */
const PLAY_FOLDERS: readonly string[] = ['components/play', 'components/looks', 'components/mascot', 'lib/play', 'lib/looks', 'lib/mascot']
const SOURCE_FILE = /\.(ts|tsx)$/
const TEST_FILE = /\.test\.tsx?$/
const TEST_DIR = '__tests__'

/** What Play's folders may never import: the core by alias, Next, the core's data and toast libraries, its stores. */
const BANNED: readonly RegExp[] = [
  /^@\//,
  /^next(\/|$)/,
  /^@tanstack\/react-query(\/|$)/,
  /^sonner(\/|$)/,
  /^zustand(\/|$)/,
  /api-client/,
]

/** Files the scan must find, so a moved folder can never pass by scanning nothing. */
const MUST_FIND: readonly string[] = [
  'components/play/chat/play-chat.tsx',
  'components/looks/looks-screen.tsx',
  'components/mascot/auto-mascot.tsx',
  'lib/play/shelf.ts',
  'lib/looks/engine.ts',
  'lib/mascot/data-events.ts',
  'lib/mascot/events.ts',
  'lib/mascot/ring.ts',
  'lib/mascot/types.ts',
]
/** Imports the scan must see, so a broken reader can never pass by finding nothing. */
const MUST_SEE: readonly string[] = ['@automatos/theme-sdk', 'react', '../look-kit/css', '../looks/engine']

/** The host code W-A5 moved out of lib/looks: where it was, and where it is. */
const MOVED_OUT: ReadonlyArray<readonly [string, string]> = [
  ['lib/looks/prepaint.ts', 'lib/theme-host/builtin/play-prepaint.ts'],
  ['lib/looks/storage.ts', 'lib/theme-host/builtin/play-look-storage.ts'],
  ['lib/looks/store.ts', 'lib/theme-host/builtin/play-look-state.ts'],
]

function runtimeFilesUnder(dir: string): string[] {
  return readdirSync(path.join(FRONTEND, dir)).flatMap((entry) => {
    const rel = `${dir}/${entry}`
    if (statSync(path.join(FRONTEND, rel)).isDirectory()) return entry === TEST_DIR ? [] : runtimeFilesUnder(rel)
    return SOURCE_FILE.test(entry) && !TEST_FILE.test(entry) ? [rel] : []
  })
}

function specifiersIn(file: string): string[] {
  return moduleSpecifiers(file, readFileSync(path.join(FRONTEND, file), 'utf8'))
}

function isBanned(specifier: string): boolean {
  return BANNED.some((pattern) => pattern.test(specifier))
}

function bannedIn(file: string): string[] {
  return specifiersIn(file).filter(isBanned).map((specifier) => `${file}: '${specifier}'`)
}

const FILES: readonly string[] = PLAY_FOLDERS.flatMap(runtimeFilesUnder)

describe("PRD-261 W-A5 — the scan of Play's six folders", () => {
  it('walks every folder, finds the files it must, and sees the imports it must', () => {
    expect(PLAY_FOLDERS.filter((folder) => !FILES.some((file) => file.startsWith(`${folder}/`)))).toEqual([])
    expect(MUST_FIND.filter((file) => !FILES.includes(file))).toEqual([])
    expect(FILES.filter((file) => file.includes(`/${TEST_DIR}/`) || TEST_FILE.test(file))).toEqual([])
    const seen = new Set(FILES.flatMap(specifiersIn))
    expect(MUST_SEE.filter((specifier) => !seen.has(specifier))).toEqual([])
  })

  it('refuses every banned kind, and lets the SDK, React and relative paths through', () => {
    const refused = ['@/lib/chat/receipts', '@/types', 'next/navigation', 'next', '@tanstack/react-query', 'sonner', 'zustand', 'zustand/middleware', '../api-client']
    expect(refused.filter((specifier) => !isBanned(specifier))).toEqual([])
    const allowed = ['@automatos/theme-sdk', 'react', 'react-dom', 'lucide-react', '../look-kit/css', './types', 'next-themes']
    expect(allowed.filter(isBanned)).toEqual([])
  })

  it('reads specifiers through the compiler, so comments and strings never count and type imports do', () => {
    const source = "// import x from '@/lib/cookies'\nconst s = \"from '@/types'\"\nimport type { R } from '@/types'\n"
    expect(moduleSpecifiers('sample.ts', source)).toEqual(['@/types'])
  })
})

describe("PRD-261 W-A5 — Play's folders import nothing by the core's alias", () => {
  it('no runtime file of the six folders names @/…, next/*, react-query, sonner, zustand or api-client', () => {
    expect(FILES.flatMap(bannedIn)).toEqual([])
  })

  it("keeps the host code W-A5 moved out of Play's folders out, with no copy left behind", () => {
    expect(MOVED_OUT.filter(([from]) => existsSync(path.join(FRONTEND, from))).map(([from]) => from)).toEqual([])
    expect(MOVED_OUT.filter(([, to]) => !existsSync(path.join(FRONTEND, to))).map(([, to]) => to)).toEqual([])
  })
})
