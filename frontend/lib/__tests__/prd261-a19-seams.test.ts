// @vitest-environment node
/**
 * PRD-261 A-19a — the core's own code leaves Play's folders, so deleting builtin Play (W-A8) cannot
 * break the SDK's chat (useChatController, useStartChat), the history refresh or the look scope.
 *
 * - No non-test file outside Play's code imports it. Play's code is its six folders
 *   (components/{play,looks,mascot}, lib/{play,looks,mascot}), its hooks (hooks/use-{shelf-mascot,
 *   celebrate-new,play-*,auto-mascot,backend-mascot,mascot-player}) and its stores (stores/play-*).
 *   hooks/use-media-preferences.ts is the core's (lib/theme-sdk/context.ts reads it), not Play's.
 *   The builtin seam may import Play: lib/theme-host/builtin/**, app/layout.tsx, app/looks/**,
 *   components/providers.tsx, components/theme-host/{theme-look-runtime,theme-slot-registrations}.tsx
 *   and lib/theme-host/boot.ts. PENDING names core code that still does and is not A-19a's, each
 *   with why; it must still import Play, so the list never outlives its reason.
 * - The seams A-19a made: lib/chat/chat-status.ts (lib/mascot/events.ts re-exports it),
 *   lib/theme-host/focus.ts (builtin Play keeps an equal copy, lib/play/focus.ts: its code may import
 *   only Play's folders, prd261-port-conversations.test.ts), stores/fresh-draft-store.ts (git mv of
 *   stores/play-chat-store.ts) and lib/look-kit/scope.ts (git mv of lib/looks/scope.ts, which
 *   re-exports it), and each importer A-19a switched names its new seam.
 *
 * Specifiers are read through the TypeScript compiler (./module-references), so a comment or a
 * string never counts and type imports and vi.mock() ids do.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import { NO_REPLY as CORE_NO_REPLY } from '@/lib/chat/chat-status'
import * as hostScope from '@/lib/look-kit/scope'
import * as playScope from '@/lib/looks/scope'
import { NO_REPLY as PLAY_NO_REPLY } from '@/lib/mascot/events'

import { moduleSpecifiers, parseSource } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
const SKIP_DIRS: ReadonlySet<string> = new Set(['node_modules', 'public'])
const SOURCE_FILE = /\.(ts|tsx)$/
const TEST_PATH = /(^|\/)__tests__\/|\.(test|spec)\.tsx?$/
/** Parsing the frontend's sources takes a few seconds; the default 5 s is too tight on a busy runner. */
const SCAN_TIMEOUT_MS = 120_000

/** Play's code, relative to the frontend. */
const PLAY_DIRS: readonly string[] = ['components/play', 'components/looks', 'components/mascot', 'lib/play', 'lib/looks', 'lib/mascot']
const PLAY_HOOK = /^hooks\/use-(shelf-mascot|celebrate-new|play-[\w-]+|auto-mascot|backend-mascot|mascot-player)(\.tsx?)?$/
const PLAY_STORE = /^stores\/play-[\w-]+(\.ts)?$/
/** A specifier naming Play's code holds one of these words, so a file holding none is not parsed. */
const MAY_NAME_PLAY = /play|looks|mascot|celebrate/

/** The builtin seam: where the core registers builtin Play, so it may import Play's code. */
const SEAM_DIRS: readonly string[] = ['lib/theme-host/builtin', 'app/looks']
const SEAM_FILES: readonly string[] = [
  'app/layout.tsx',
  'components/providers.tsx',
  'components/theme-host/theme-look-runtime.tsx',
  'components/theme-host/theme-slot-registrations.tsx',
  'lib/theme-host/boot.ts',
]
/** Core code that still imports Play's and is not A-19a's to move, with why. */
const PENDING: Readonly<Record<string, string>> = {
  'hooks/use-media-preferences.ts':
    "useVideoFormat reads Play's lib/mascot/media (videoFormat); A-14e's mascot guard holds the hook's relative imports to Play's mascot code",
}

/** Each importer A-19a switched, and the seam it names now. */
const SWITCHED: ReadonlyArray<readonly [string, string]> = [
  ['lib/chat/message-text.ts', '@/lib/chat/chat-status'],
  ['hooks/use-refresh-history-on-turn-end.ts', '@/lib/chat/chat-status'],
  ['components/theme-host/theme-slot-page.tsx', '@/lib/theme-host/focus'],
  ['lib/theme-sdk/chat.ts', '@/stores/fresh-draft-store'],
  ['hooks/use-start-chat.ts', '@/stores/fresh-draft-store'],
  ['components/theme-host/account-look-scope.tsx', '@/lib/look-kit/scope'],
  ['components/theme-host/look-scope.ts', '@/lib/look-kit/scope'],
  ['lib/theme-host/builtin/play-look-storage.ts', '@/lib/look-kit/scope'],
  ['lib/theme-host/builtin/play-look-store.ts', '@/lib/look-kit/scope'],
]
/** The functions builtin Play's lib/play/focus.ts copies from the host's lib/theme-host/focus.ts. */
const FOCUS_COPY: readonly string[] = ['focusWasDropped', 'refocus', 'refocusWhenShown']
/** A Play import through the builtin seam the scan must see, so a broken reader can never pass. */
const MUST_SEE = 'lib/theme-host/builtin/play-slots.tsx: @/components/play/conversations/play-conversations-lazy'

interface Found {
  readonly file: string
  readonly specifier: string
}

function sourcesUnder(dir: string): string[] {
  return readdirSync(path.join(FRONTEND, dir)).flatMap((entry) => {
    const rel = dir === '' ? entry : `${dir}/${entry}`
    if (statSync(path.join(FRONTEND, rel)).isDirectory()) {
      return SKIP_DIRS.has(entry) || entry.startsWith('.') ? [] : sourcesUnder(rel)
    }
    return SOURCE_FILE.test(entry) ? [rel] : []
  })
}

/** The frontend path a specifier written in `file` names; null for a package. */
function targetOf(file: string, specifier: string): string | null {
  if (specifier.startsWith('@/')) return specifier.slice(2)
  if (specifier.startsWith('.')) return path.posix.normalize(path.posix.join(path.posix.dirname(file), specifier))
  return null
}

function isPlayCode(target: string): boolean {
  const inFolder = PLAY_DIRS.some((dir) => target === dir || target.startsWith(`${dir}/`))
  return inFolder || PLAY_HOOK.test(target) || PLAY_STORE.test(target)
}

function isSeam(file: string): boolean {
  return SEAM_FILES.includes(file) || SEAM_DIRS.some((dir) => file.startsWith(`${dir}/`))
}

function source(file: string): string {
  return readFileSync(path.join(FRONTEND, file), 'utf8')
}

function specifiersOf(file: string): string[] {
  return moduleSpecifiers(file, source(file))
}

/** The imports of Play's code in `file`. */
function playImportsOf(file: string): Found[] {
  const text = source(file)
  if (!MAY_NAME_PLAY.test(text)) return []
  return moduleSpecifiers(file, text)
    .filter((specifier) => {
      const target = targetOf(file, specifier)
      return target !== null && isPlayCode(target)
    })
    .map((specifier) => ({ file, specifier }))
}

const describeFound = ({ file, specifier }: Found): string => `${file}: ${specifier}`

/** A function declaration's own text (its doc comment left out), or null when the file has none by that name. */
function functionText(file: string, name: string): string | null {
  const sourceFile = parseSource(file, source(file))
  const found = sourceFile.statements.find(
    (statement): statement is ts.FunctionDeclaration => ts.isFunctionDeclaration(statement) && statement.name?.text === name,
  )
  return found === undefined ? null : found.getText(sourceFile)
}

const CORE_FILES: readonly string[] = sourcesUnder('').filter((file) => !TEST_PATH.test(file) && !isPlayCode(file))

describe("PRD-261 A-19a — the scan of the core's imports of Play", () => {
  it('reads the frontend, its builtin seam, and the Play imports it must see', () => {
    const found = CORE_FILES.flatMap(playImportsOf).map(describeFound)
    expect(found).toContain(MUST_SEE)
    expect(SWITCHED.map(([file]) => file).filter((file) => !CORE_FILES.includes(file))).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it("tells Play's code from the core's", () => {
    const play = ['components/play/chat/play-chat', 'lib/looks/scope', 'lib/mascot/events', 'hooks/use-play-turn', 'hooks/use-auto-mascot.ts', 'stores/play-chat-store']
    const core = ['components/playbooks/list', 'hooks/use-playbook-api', 'hooks/use-media-preferences', 'stores/fresh-draft-store', 'lib/look-kit/scope', 'lib/chat/chat-status', 'lib/theme-host/focus']
    expect(play.filter((target) => !isPlayCode(target))).toEqual([])
    expect(core.filter(isPlayCode)).toEqual([])
    expect(targetOf('components/theme-host/theme-slot-page.tsx', '../../lib/play/focus')).toBe('lib/play/focus')
    expect(targetOf('lib/chat/message-text.ts', '@/lib/mascot/events')).toBe('lib/mascot/events')
    expect(targetOf('lib/chat/message-text.ts', '@automatos/theme-sdk')).toBeNull()
  })

  it('names only seam and pending files that exist', () => {
    const named = [...SEAM_DIRS, ...SEAM_FILES, ...Object.keys(PENDING)]
    expect(named.filter((file) => !existsSync(path.join(FRONTEND, file)))).toEqual([])
  })
})

describe("PRD-261 A-19a — the core's code imports Play's only through the builtin seam", () => {
  it("no non-test file outside Play's code imports it, but the builtin seam and the pending list", () => {
    const found = CORE_FILES.flatMap(playImportsOf).filter(({ file }) => !isSeam(file) && PENDING[file] === undefined)
    expect(found.map(describeFound)).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it('keeps no pending entry past its reason: each still imports Play', () => {
    const stale = Object.keys(PENDING).filter((file) => playImportsOf(file).length === 0)
    expect(stale).toEqual([])
  })

  it('switched every importer A-19a names to its new seam', () => {
    const missing = SWITCHED.filter(([file, seam]) => !specifiersOf(file).includes(seam))
    expect(missing.map(([file, seam]) => `${file}: ${seam}`)).toEqual([])
  })
})

describe('PRD-261 A-19a — the seams it made', () => {
  it('moved the fresh-draft store and the look scope (git mv), leaving no copy of the store', () => {
    expect(existsSync(path.join(FRONTEND, 'stores/play-chat-store.ts'))).toBe(false)
    expect(existsSync(path.join(FRONTEND, 'stores/fresh-draft-store.ts'))).toBe(true)
    expect(existsSync(path.join(FRONTEND, 'lib/look-kit/scope.ts'))).toBe(true)
  })

  it("re-exports the host's look scope at Play's old path, and nothing else", () => {
    const file = 'lib/looks/scope.ts'
    const sourceFile = parseSource(file, source(file))
    const reexports = sourceFile.statements.filter((statement) => ts.isExportDeclaration(statement))
    expect(reexports).toHaveLength(sourceFile.statements.length)
    expect(specifiersOf(file)).toEqual(['../look-kit/scope'])
    const exported = Object.keys(hostScope) as Array<keyof typeof hostScope>
    expect(exported.filter((name) => playScope[name] !== hostScope[name])).toEqual([])
  })

  it("hands Play's mascot events the core's NO_REPLY, the same object", () => {
    expect(PLAY_NO_REPLY).toBe(CORE_NO_REPLY)
  })

  it("keeps builtin Play's focus helpers equal to the host's", () => {
    const host = FOCUS_COPY.map((name) => functionText('lib/theme-host/focus.ts', name))
    expect(host.filter((text) => text === null)).toEqual([])
    expect(FOCUS_COPY.map((name) => functionText('lib/play/focus.ts', name))).toEqual(host)
  })

  it('keeps hooks/use-media-preferences.ts in core, read by the SDK context', () => {
    expect(isPlayCode('hooks/use-media-preferences.ts')).toBe(false)
    expect(specifiersOf('lib/theme-sdk/context.ts')).toContain('@/hooks/use-media-preferences')
  })
})
