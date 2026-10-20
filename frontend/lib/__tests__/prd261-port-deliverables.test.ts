/**
 * PRD-261 A-14b (W-A5, lane L2) — Play's deliverables shelf reaches the core only through the
 * theme SDK, so W-K2 can copy it into the Design Kit as part of an installable package.
 *
 * A static scan of exactly the area (plan §2.4: components/play/deliverables/**,
 * hooks/use-shelf-mascot.ts, hooks/use-celebrate-new.ts, lib/play/{shelf,seen}.ts; runtime files,
 * not their tests), read through the TypeScript compiler (./module-references.ts), so a comment
 * or a string never counts and a type-only import does:
 *
 * - nothing on W-A5's ban list: `@/…`, `next/*`, `@tanstack/react-query`, `sonner`, the backend
 *   client, zustand (the area keeps no store);
 * - packages only from the allowed list: `@automatos/theme-sdk`, `react`, `react-dom`;
 * - every relative import lands inside Play's own folders (its components, lib, mascot library,
 *   looks, and the hooks and stores that move with it), so no `../../lib/<core>` slips past the ban;
 * - the seam items (a core import the SDK has no name for yet, PA-16) are allowed one file and one
 *   import each, and the allowance fails once its import has gone.
 *
 * And the pure utilities and wording the area copied from the core (sdk-surface-map.md: "copied
 * into Play") stay equal to the core's own.
 */
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import {
  deliverableHref as coreDeliverableHref,
  isDeliverableId as coreIsDeliverableId,
  UNTITLED as CORE_UNTITLED,
} from '@/lib/chat/deliverables'
import { DOWNLOAD_FAILED as CORE_DOWNLOAD_FAILED } from '@/lib/deliverables/download'
import { deliverableHref, DOWNLOAD_FAILED, isDeliverableId, UNTITLED } from '@/lib/play/shelf'

import { moduleSpecifiers } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
/** The area A-14b ports: its folder, and the hooks and lib files that belong to it. */
const AREA_DIRS = ['components/play/deliverables']
const AREA_FILES = ['hooks/use-shelf-mascot.ts', 'hooks/use-celebrate-new.ts', 'lib/play/shelf.ts', 'lib/play/seen.ts']
const SOURCE_FILE = /\.(ts|tsx)$/
const TEST_FILE = /\.test\.tsx?$/
const TEST_DIR = '__tests__'

/** The packages a ported file may import: the SDK, and React (the host's, through the import map). */
const ALLOWED_PACKAGES: ReadonlySet<string> = new Set(['@automatos/theme-sdk', 'react', 'react-dom'])
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
/**
 * Seam items: [file, import] for a core import the SDK has no name for yet. Each file says why.
 * socials-offered.ts: whether the platform offers Socials; `context.useWorkspace` doesn't carry it.
 * use-celebrate-new.ts: whose seen record it is (the signed-in user in the workspace, PRD-260 F5);
 * the host fills that context (components/theme-host/look-scope.ts), and no SDK name carries the user.
 */
const PENDING: ReadonlyArray<readonly [string, string]> = [
  ['components/play/deliverables/socials-offered.ts', '@/components/workspace-provider'],
  ['hooks/use-celebrate-new.ts', '@/components/theme-host/look-scope'],
]
/** Imports the scan must see, so a broken reader can never pass by finding nothing. */
const MUST_SEE = ['@automatos/theme-sdk', 'react']
/** Files the scan must find, so a moved folder can never pass by scanning nothing. */
const MUST_FIND = ['components/play/deliverables/play-shelf.tsx', 'components/play/deliverables/shelf-card.tsx', ...AREA_FILES]

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

function areaFiles(): string[] {
  const inDirs = AREA_DIRS.flatMap((dir) => runtimeFilesUnder(path.join(FRONTEND, dir)))
  return [...inDirs, ...AREA_FILES.map((file) => path.join(FRONTEND, file))].map(relative).sort()
}

function importsOf(file: string): FoundImport[] {
  const full = path.join(FRONTEND, file)
  return moduleSpecifiers(full, fs.readFileSync(full, 'utf8')).map((specifier) => ({ file, specifier }))
}

function isRelative(specifier: string): boolean {
  return specifier.startsWith('.')
}

function isBanned(specifier: string): boolean {
  return BANNED.some((pattern) => pattern.test(specifier))
}

function isPending({ file, specifier }: FoundImport): boolean {
  return PENDING.some(([pendingFile, pendingImport]) => pendingFile === file && pendingImport === specifier)
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

describe('PRD-261 A-14b — the scan of the deliverables area', () => {
  it('finds the area and reads its imports', () => {
    expect(MUST_FIND.filter((file) => !FILES.includes(file))).toEqual([])
    const specifiers = new Set(IMPORTS.map((found) => found.specifier))
    expect(MUST_SEE.filter((specifier) => !specifiers.has(specifier))).toEqual([])
  })

  it('imports nothing on the ban list but its seam items', () => {
    const banned = IMPORTS.filter((found) => isBanned(found.specifier) && !isPending(found))
    expect(banned.map(describeImport)).toEqual([])
  })

  it('imports packages only from the allowed list', () => {
    const packages = IMPORTS.filter(({ specifier }) => !isRelative(specifier) && !isBanned(specifier))
    expect(packages.filter(({ specifier }) => !ALLOWED_PACKAGES.has(specifier)).map(describeImport)).toEqual([])
  })

  it("lands every relative import inside Play's own folders", () => {
    const outside = IMPORTS.filter((found) => isRelative(found.specifier) && !landsInPlay(found))
    expect(outside.map((found) => `${describeImport(found)} -> ${landing(found)}`)).toEqual([])
  })

  it('keeps each seam item only while its import is there', () => {
    const found = new Set(IMPORTS.map(describeImport))
    const stale = PENDING.map(([file, specifier]) => describeImport({ file, specifier })).filter((item) => !found.has(item))
    expect(stale).toEqual([])
  })
})

describe('PRD-261 A-14b — the scan helpers', () => {
  it('bans the core by path, Next, react-query, sonner, the backend client and zustand', () => {
    const banned = ['@/lib/ticket-links', 'next/link', 'next', '@tanstack/react-query', 'sonner', '@/lib/api-client', 'zustand', 'zustand/middleware']
    expect(banned.filter((specifier) => !isBanned(specifier))).toEqual([])
    expect(['@automatos/theme-sdk', 'react', './shelf-card', 'nextra'].filter(isBanned)).toEqual([])
  })

  it("tells a relative import that stays in Play's folders from one that leaves them", () => {
    const from = 'components/play/deliverables/play-shelf.tsx'
    const inside = ['../../mascot/auto-mascot', '../shell/page-columns', '../../../lib/play/shelf', '../../../hooks/use-shelf-mascot', './shelf.module.css']
    const outside = ['../../../lib/chat/deliverables', '../../../hooks/use-deliverables-api', '../../layout/main-layout']
    expect(inside.filter((specifier) => !landsInPlay({ file: from, specifier }))).toEqual([])
    expect(outside.filter((specifier) => landsInPlay({ file: from, specifier }))).toEqual([])
  })
})

describe('PRD-261 A-14b — what the area copied from the core stays the same', () => {
  const ids = ['d-laptop', 'A_b-9', 'x'.repeat(128), 'x'.repeat(129), '', '../../admin?x=1', 'd laptop', 'd/1']

  it("checks a Deliverable id with the core's pattern", () => {
    expect(ids.map(isDeliverableId)).toEqual(ids.map(coreIsDeliverableId))
  })

  it("links a Deliverable to the core's page for it", () => {
    for (const id of ['d-laptop', 'A_b-9']) expect(deliverableHref(id)).toBe(coreDeliverableHref(id))
  })

  it("names an untitled Deliverable and a failed download in the core's words", () => {
    expect(UNTITLED).toBe(CORE_UNTITLED)
    expect(DOWNLOAD_FAILED).toBe(CORE_DOWNLOAD_FAILED)
  })
})
