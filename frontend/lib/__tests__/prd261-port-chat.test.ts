/**
 * PRD-261 A-14a (W-A5, lane L1) — Play's chat reaches the core only through the theme SDK, so W-K2
 * can copy it into the Design Kit as part of an installable package.
 *
 * A static scan of exactly the area (plan §2.4: components/play/chat/**, hooks/use-play-conversation.ts,
 * hooks/use-play-turn.ts, stores/play-*; runtime files, not their tests), read through the TypeScript
 * compiler (./module-references.ts), so a comment or a string never counts and a type-only import does:
 *
 * - nothing on W-A5's ban list: `@/…`, `next/*`, `@tanstack/react-query`, `sonner`, the backend
 *   client, and zustand outside Play's own stores (stores/play-* keep their `create`);
 * - packages only from the allowed list: `@automatos/theme-sdk`, `react`, `react-dom`, and `zustand`
 *   in Play's own stores;
 * - every relative import lands inside Play's own folders (its components, lib, mascot library,
 *   looks, and the hooks and stores that move with it), so no `../../lib/<core>` slips past the ban.
 *   Nothing is pending: the area's one seam item, the host's AI label fence, is the SDK's
 *   `cards.AiLabelFence` since W-A5's seam review (sdk-surface-map.md).
 *
 * And what the area copied from the core (sdk-surface-map.md: "copied into Play") stays equal to the
 * core's own: the receipts' closing notes, a Deliverable's id check, link and type words, the Mission's
 * context size, and the readings of the conversation (the newest reply, the reply being written as
 * Auto reads it, its running tool's input, the recent messages), each on the SDK's messages against the
 * core's on the core's.
 */
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import {
  deliverableHref as coreDeliverableHref,
  isDeliverableId as coreIsDeliverableId,
  KIND_WORDS as CORE_KIND_WORDS,
  mergeDeliverables as coreMergeDeliverables,
} from '@/lib/chat/deliverables'
import {
  latestReply as coreLatestReply,
  replyProgress as coreReplyProgress,
  runningToolInput as coreRunningToolInput,
} from '@/lib/chat/message-text'
import { NO_ACTIONS as CORE_NO_ACTIONS, NOTHING_ON_THE_BOARD as CORE_NOTHING_ON_THE_BOARD } from '@/lib/chat/receipts'
import { CONTEXT_MESSAGES_KEPT as CORE_CONTEXT_KEPT, recentMessages as coreRecentMessages } from '@/lib/chat/suggestion'
import { deliverableHref, isDeliverableId, KIND_WORDS } from '@/components/play/chat/deliverable-link'
import { NO_ACTIONS, NOTHING_ON_THE_BOARD } from '@/components/play/chat/receipts'
import {
  CONTEXT_MESSAGES_KEPT,
  latestReply,
  liveMessages,
  mergeDeliverables,
  recentMessages,
  replyProgress,
  runningToolInput,
} from '@/components/play/chat/reply'
import type { ChatConversationView } from '@/lib/chat-controller/view-model'
import { type ControllerReading, hostMessages } from '@/lib/theme-sdk/internal/chat-view'
import type { ChatMessage } from '@/types'

import { moduleSpecifiers } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
/** The area A-14a ports: its folder, the two hooks, and Play's own stores (stores/play-*). */
const AREA_DIRS = ['components/play/chat']
const AREA_HOOKS = ['hooks/use-play-conversation.ts', 'hooks/use-play-turn.ts']
const STORES_DIR = 'stores'
const PLAY_STORE = /^play-[\w-]+\.ts$/
const SOURCE_FILE = /\.(ts|tsx)$/
const TEST_FILE = /\.test\.tsx?$/
const TEST_DIR = '__tests__'

/** The packages a ported file may import: the SDK, and React (the host's, through the import map). */
const ALLOWED_PACKAGES: ReadonlySet<string> = new Set(['@automatos/theme-sdk', 'react', 'react-dom'])
/** Play's own stores keep their zustand `create` (plan §2.6 W-A5); nothing else in the area may. */
const STORE_PACKAGE = 'zustand'
/** W-A5's ban list. */
const BANNED: readonly RegExp[] = [
  /^@\//,
  /^next(\/|$)/,
  /^@tanstack\/react-query(\/|$)/,
  /^sonner(\/|$)/,
  /api-client/,
  /^zustand(\/|$)/,
]
/** Play's own folders, relative to the frontend: where a relative import may land. */
const PLAY_HOME: readonly RegExp[] = [
  /^components\/(play|looks|mascot)\//,
  /^lib\/(play|looks|mascot)\//,
  /^hooks\/use-(shelf-mascot|celebrate-new|play-[\w-]+|auto-mascot|backend-mascot|mascot-player|media-preferences)(\.tsx?)?$/,
  /^stores\/play-[\w-]+(\.ts)?$/,
]
/** Imports the scan must see, so a broken reader can never pass by finding nothing. */
const MUST_SEE = ['@automatos/theme-sdk', 'react', STORE_PACKAGE]
/** Files the scan must find, so a moved folder can never pass by scanning nothing. */
const MUST_FIND = [
  'components/play/chat/play-chat.tsx',
  'components/play/chat/play-chat-view.tsx',
  'components/play/chat/auto-reply.tsx',
  'stores/play-chat-store.ts',
  ...AREA_HOOKS,
]

interface FoundImport {
  readonly file: string
  readonly specifier: string
}

function relative(file: string): string {
  return path.relative(FRONTEND, file).split(path.sep).join('/')
}

function runtimeFilesUnder(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return entry.name === TEST_DIR ? [] : runtimeFilesUnder(full)
    return SOURCE_FILE.test(entry.name) && !TEST_FILE.test(entry.name) ? [full] : []
  })
}

function playStores(): string[] {
  const dir = path.join(FRONTEND, STORES_DIR)
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && PLAY_STORE.test(entry.name))
    .map((entry) => path.join(dir, entry.name))
}

function areaFiles(): string[] {
  const inDirs = AREA_DIRS.flatMap((dir) => runtimeFilesUnder(path.join(FRONTEND, dir)))
  const hooks = AREA_HOOKS.map((file) => path.join(FRONTEND, file))
  return [...inDirs, ...hooks, ...playStores()].map(relative).sort()
}

function importsOf(file: string): FoundImport[] {
  const full = path.join(FRONTEND, file)
  return moduleSpecifiers(full, fs.readFileSync(full, 'utf8')).map((specifier) => ({ file, specifier }))
}

function isRelative(specifier: string): boolean {
  return specifier.startsWith('.')
}

function isPlayStore(file: string): boolean {
  return file.startsWith(`${STORES_DIR}/`) && PLAY_STORE.test(path.posix.basename(file))
}

/** A store of Play's own importing zustand itself is allowed; zustand's middleware and anything else is not. */
function isAllowedStoreImport({ file, specifier }: FoundImport): boolean {
  return specifier === STORE_PACKAGE && isPlayStore(file)
}

function isBanned(found: FoundImport): boolean {
  return BANNED.some((pattern) => pattern.test(found.specifier)) && !isAllowedStoreImport(found)
}

/** Where a relative import lands, relative to the frontend. */
function landing({ file, specifier }: FoundImport): string {
  return path.posix.normalize(path.posix.join(path.posix.dirname(file), specifier))
}

function landsInPlay(found: FoundImport): boolean {
  const target = landing(found)
  return PLAY_HOME.some((pattern) => pattern.test(target))
}

function describeImport({ file, specifier }: FoundImport): string {
  return `${file}: ${specifier}`
}

const FILES = areaFiles()
const IMPORTS = FILES.flatMap(importsOf)

describe('PRD-261 A-14a — the scan of the chat area', () => {
  it('finds the area and reads its imports', () => {
    expect(MUST_FIND.filter((file) => !FILES.includes(file))).toEqual([])
    const specifiers = new Set(IMPORTS.map((found) => found.specifier))
    expect(MUST_SEE.filter((specifier) => !specifiers.has(specifier))).toEqual([])
  })

  it('imports nothing on the ban list (but zustand in Play’s own stores)', () => {
    const banned = IMPORTS.filter(isBanned)
    expect(banned.map(describeImport)).toEqual([])
  })

  it('imports packages only from the allowed list', () => {
    const packages = IMPORTS.filter((found) => !isRelative(found.specifier) && !isBanned(found) && !isAllowedStoreImport(found))
    expect(packages.filter(({ specifier }) => !ALLOWED_PACKAGES.has(specifier)).map(describeImport)).toEqual([])
  })

  it("lands every relative import inside Play's own folders", () => {
    const outside = IMPORTS.filter((found) => isRelative(found.specifier) && !landsInPlay(found))
    expect(outside.map((found) => `${describeImport(found)} -> ${landing(found)}`)).toEqual([])
  })
})

describe('PRD-261 A-14a — the scan helpers', () => {
  const chat = 'components/play/chat/play-chat-view.tsx'

  it('bans the core by path, Next, react-query, sonner, the backend client and zustand outside Play’s stores', () => {
    const banned = ['@/lib/ticket-links', 'next/link', 'next/dynamic', 'next', '@tanstack/react-query', 'sonner', '@/lib/api-client', 'zustand']
    expect(banned.filter((specifier) => !isBanned({ file: chat, specifier }))).toEqual([])
    expect(['@automatos/theme-sdk', 'react', './reply', 'nextra'].filter((specifier) => isBanned({ file: chat, specifier }))).toEqual([])
  })

  it("allows zustand in Play's own stores only, and never its middleware", () => {
    expect(isBanned({ file: 'stores/play-chat-store.ts', specifier: STORE_PACKAGE })).toBe(false)
    expect(isBanned({ file: 'stores/play-chat-store.ts', specifier: 'zustand/middleware' })).toBe(true)
    expect(isBanned({ file: 'stores/chat-session-store.ts', specifier: STORE_PACKAGE })).toBe(true)
    expect(isBanned({ file: 'hooks/use-play-turn.ts', specifier: STORE_PACKAGE })).toBe(true)
  })

  it("tells a relative import that stays in Play's folders from one that leaves them", () => {
    const inside = ['../../mascot/auto-mascot', '../shell/page-columns', '../../../lib/mascot/events', '../../../hooks/use-play-turn', './chat.module.css']
    const outside = ['../../../lib/chat/message-text', '../../../hooks/use-backend-health', '../../layout/main-layout', '../../ai-label/ai-label-fence']
    expect(inside.filter((specifier) => !landsInPlay({ file: chat, specifier }))).toEqual([])
    expect(outside.filter((specifier) => landsInPlay({ file: chat, specifier }))).toEqual([])
  })
})

// --- what the area copied from the core ------------------------------------------------------------

const message = (id: string, role: 'user' | 'assistant', text: string, extra: Record<string, unknown> = {}) =>
  ({ id, role, content: text, parts: [{ type: 'text', text }], ...extra }) as unknown as ChatMessage

const NO_APPROVALS = { of: () => [], loose: [], answered: () => undefined, closed: () => undefined, waiting: false }
const VIEW = { key: 'c1', chatId: 'c1', title: null, messages: [], awaitingReply: false } as unknown as ChatConversationView

/** The core's messages as the SDK hands them to a theme (lib/theme-sdk/internal/chat-view.ts). */
function shown(messages: readonly ChatMessage[], stopped: ReadonlySet<string> = new Set()) {
  const reading = {
    messages,
    streaming: false,
    isStopped: (id: string) => stopped.has(id),
    deliverablesOf: () => [],
    approvals: NO_APPROVALS,
  } as unknown as ControllerReading
  return hostMessages(reading, VIEW).list
}

const ASK = message('u1', 'user', 'Shortlist three laptops?')
const SEARCHING = message('r1', 'assistant', '', {
  toolCalls: [
    { toolCallId: 't1', toolName: 'platform_web_search', state: 'completed', input: { query: 'old' } },
    { toolCallId: 't2', toolName: 'platform_web_search', state: 'running', input: { query: 'budget laptops' } },
  ],
  reasoning: 'Thinking about budgets',
})
const ANSWERED = message('r1', 'assistant', 'Here are three', { limitReached: { limit: 'turns', value: 3, message: 'Turn limit reached.' } })
const BLANK_CAP = message('r1', 'assistant', '  ', { limitReached: { limit: 'turns', value: 3, message: '' } })
const CONVERSATIONS: ReadonlyArray<readonly ChatMessage[]> = [
  [],
  [ASK],
  [ASK, SEARCHING],
  [ASK, ANSWERED],
  [ASK, BLANK_CAP],
  [ASK, ANSWERED, message('u2', 'user', 'And the battery?')],
  [message('u0', 'user', 'a'), message('r0', 'assistant', 'b'), ASK, ANSWERED, message('u2', 'user', 'c'), SEARCHING],
]

describe('PRD-261 A-14a — what the chat area copied from the core stays the same', () => {
  it("closes the receipts in the core's words", () => {
    expect(NO_ACTIONS).toBe(CORE_NO_ACTIONS)
    expect(NOTHING_ON_THE_BOARD).toBe(CORE_NOTHING_ON_THE_BOARD)
  })

  it("checks a Deliverable id with the core's pattern, links it to the core's page and names its type in the core's words", () => {
    const ids = ['d-laptop', 'A_b-9', 'x'.repeat(128), 'x'.repeat(129), '', '../../admin?x=1', 'd laptop', 'd/1']
    expect(ids.map(isDeliverableId)).toEqual(ids.map(coreIsDeliverableId))
    for (const id of ['d-laptop', 'A_b-9', 'a b']) expect(deliverableHref(id)).toBe(coreDeliverableHref(id))
    expect(KIND_WORDS).toEqual(CORE_KIND_WORDS)
  })

  it("merges a reply's Deliverables as the core does: one per id, the first seen wins, the same list when nothing is new", () => {
    const shelf = { id: 'd-1', title: 'Shortlist', kind: 'document' } as const
    const cases = [
      [[shelf], [{ ...shelf, title: 'Again' }]],
      [[shelf], [{ id: 'd-2', title: 'Chart', kind: 'image' }, { id: 'd-2', title: 'Dup', kind: 'image' }]],
      [[], []],
    ] as const
    for (const [known, found] of cases) {
      expect(mergeDeliverables(known, found)).toEqual(coreMergeDeliverables(known, found))
    }
    const known = [shelf]
    expect(mergeDeliverables(known, [shelf])).toBe(known)
  })

  it("plans a Mission from the core's number of recent messages, in the core's shape", () => {
    expect(CONTEXT_MESSAGES_KEPT).toBe(CORE_CONTEXT_KEPT)
    for (const conversation of CONVERSATIONS) {
      expect(recentMessages(shown(conversation))).toEqual(coreRecentMessages(conversation))
    }
  })

  it('reads the newest reply, the reply being written and its running tool as the core reads them', () => {
    for (const conversation of CONVERSATIONS) {
      const host = shown(conversation)
      expect(latestReply(host)?.id).toBe(coreLatestReply(conversation)?.id)
      expect(replyProgress(host)).toEqual(coreReplyProgress(conversation))
      expect(runningToolInput(latestReply(host))).toEqual(coreRunningToolInput(coreLatestReply(conversation)))
    }
  })

  it('reads the conversation as the chat hook holds it: without the replies Stop kept, the same list when none is', () => {
    const conversation = [ASK, ANSWERED, message('u2', 'user', 'And the battery?'), message('r2', 'assistant', 'Ten')]
    const all = shown(conversation)
    expect(liveMessages(all)).toBe(all)
    const kept = shown(conversation, new Set(['r2']))
    expect(liveMessages(kept).map((reply) => reply.id)).toEqual(['u1', 'r1', 'u2'])
  })
})
