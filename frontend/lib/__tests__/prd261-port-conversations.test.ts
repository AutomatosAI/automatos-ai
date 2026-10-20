/**
 * PRD-261 A-14c (W-A5) — Play's conversations read the core only through the theme SDK, so W-K2
 * can copy them into the Design Kit as part of an installable package.
 *
 * The area is components/play/conversations/** and lib/play/{conversations,detail,draft,focus}.ts
 * (lib/play/{shelf,seen}.ts are A-14b's). Every module specifier in its runtime files (imports,
 * exports, import(), import types; read through the TypeScript compiler, never by raw text) is
 * '@automatos/theme-sdk', 'react', 'react-dom', or a relative path inside Play's own folders.
 * Nothing on the plan's ban list: '@/…', 'next/*', '@tanstack/react-query', 'sonner', zustand,
 * the backend client.
 *
 * Nothing is pending: the area's one seam item, the theme layout, is the SDK's
 * `context.useThemeLayout` since W-A5's seam review (sdk-surface-map.md).
 *
 * Tests (__tests__/, *.test.ts[x]) are host-side harnesses and are not scanned. The plan's literal
 * check (`git grep -nE "from '(@/|next/|@tanstack/react-query|sonner)|from 'zustand'" -- <area>`)
 * does not exclude them, so it prints the PRD-260 harness's two lines in
 * play-conversations.test.tsx: QueryClientProvider (the SDK's hooks need the host's provider, as
 * the SDK's own page tests mount it) and CONVERSATION_LIMIT (the core's history call it asserts).
 *
 * Play's copies of core code the surface map copies into Play (the one-line text rule, the
 * untitled name, the "Made by" words) are held to the core's here, so the two never drift.
 */
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import { MADE_BY_SOURCE } from '@/components/workspace/gallery-view/made-by'
import { UNTITLED } from '@/lib/chat/deliverables'
import { MADE_ELSEWHERE_WORDS, UNTITLED_DELIVERABLE } from '@/lib/play/detail'
import { cutDraft } from '@/lib/play/draft'
import { oneLine } from '@/lib/text/plain-line'

import { moduleSpecifiers } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
const AREA_DIRS = ['components/play/conversations']
const AREA_FILES = ['lib/play/conversations.ts', 'lib/play/detail.ts', 'lib/play/draft.ts', 'lib/play/focus.ts']
/** The modules the area must keep scanning (its page, its list, its lazy slot and the four lib files). */
const AREA_ENTRY_POINTS = [
  'components/play/conversations/conversation-list.tsx',
  'components/play/conversations/play-conversations-lazy.tsx',
  'components/play/conversations/play-conversations.tsx',
  ...AREA_FILES,
]
const SOURCE_FILE = /\.tsx?$/
const TEST_PATH = /(^|\/)__tests__\/|\.test\.tsx?$/
const SDK = '@automatos/theme-sdk'
const PACKAGES: ReadonlySet<string> = new Set([SDK, 'react', 'react-dom'])
/** Play's own folders (the surface map's scan): a relative import may land only inside these. */
const PLAY_FOLDERS = ['components/play/', 'components/looks/', 'components/mascot/', 'lib/play/', 'lib/looks/', 'lib/mascot/']
/** The plan's ban list (W-A5 checks), whatever else the allow-list says. */
const BANNED: readonly RegExp[] = [
  /^@\//,
  /^next(\/|$)/,
  /^@tanstack\/react-query(\/|$)/,
  /^sonner(\/|$)/,
  /^zustand(\/|$)/,
  /(^|\/)api-client(\.tsx?)?$/,
]
/** Code points the one-line rule is compared on: the BMP without surrogates, and the astral ranges it names. */
const ASTRAL_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x1f000, 0x1faff],
  [0x2fff0, 0x30010],
  [0xe0000, 0xe01ff],
  [0xf0000, 0xf00ff],
  [0x10fff0, 0x10ffff],
]
const BMP_END = 0xffff
const SURROGATES: readonly [number, number] = [0xd800, 0xdfff]

function walk(dir: string): string[] {
  return fs.readdirSync(path.join(FRONTEND, dir), { withFileTypes: true }).flatMap((entry) => {
    const rel = `${dir}/${entry.name}`
    return entry.isDirectory() ? walk(rel) : [rel]
  })
}

/** The area's runtime files, relative to the frontend; its tests left out. */
function areaFiles(): string[] {
  const inDirs = AREA_DIRS.flatMap(walk).filter((rel) => SOURCE_FILE.test(rel) && !TEST_PATH.test(rel))
  return [...inDirs, ...AREA_FILES].sort()
}

/** A relative specifier's target, relative to the frontend. */
function relativeTarget(file: string, specifier: string): string {
  return path.posix.normalize(path.posix.join(path.posix.dirname(file), specifier))
}

/** Why the area may not import this, or null when it may. */
function refusal(file: string, specifier: string): string | null {
  if (BANNED.some((pattern) => pattern.test(specifier))) return 'on the ban list'
  if (PACKAGES.has(specifier)) return null
  if (!specifier.startsWith('.')) return 'a package other than the SDK, react or react-dom'
  const target = relativeTarget(file, specifier)
  return PLAY_FOLDERS.some((folder) => target.startsWith(folder)) ? null : `outside Play's folders (${target})`
}

function specifiersOf(file: string): string[] {
  return moduleSpecifiers(file, fs.readFileSync(path.join(FRONTEND, file), 'utf8'))
}

function refusals(): string[] {
  return areaFiles().flatMap((file) =>
    specifiersOf(file).flatMap((specifier) => {
      const why = refusal(file, specifier)
      return why === null ? [] : [`${file} imports '${specifier}': ${why}`]
    }),
  )
}

function comparedCharacters(): string[] {
  const chars: string[] = []
  for (let code = 0; code <= BMP_END; code += 1) {
    if (code < SURROGATES[0] || code > SURROGATES[1]) chars.push(String.fromCodePoint(code))
  }
  for (const [low, high] of ASTRAL_RANGES) {
    for (let code = low; code <= high; code += 1) chars.push(String.fromCodePoint(code))
  }
  return chars
}

describe('PRD-261 A-14c — the guard reads what it should', () => {
  it("scans the area's runtime files and none of its tests", () => {
    const files = areaFiles()
    expect(files).toEqual(expect.arrayContaining(AREA_ENTRY_POINTS))
    expect(files.filter((rel) => TEST_PATH.test(rel))).toEqual([])
  })

  it('refuses the ban list and any package or path outside Play, and allows the SDK, React and Play', () => {
    const file = 'components/play/conversations/sample.tsx'
    const refused = [
      'next/link',
      'next/dynamic',
      '@/lib/ticket-links',
      'sonner',
      'zustand',
      '@tanstack/react-query',
      '../../../lib/api-client',
      'lucide-react',
      '../../workspace/gallery-view/made-by',
    ]
    const allowed = [
      SDK,
      'react',
      'react-dom',
      './conversations.module.css',
      '../shell/page-columns',
      '../../mascot/auto-mascot',
      '../../../lib/play/conversations',
    ]
    expect(refused.filter((specifier) => refusal(file, specifier) === null)).toEqual([])
    expect(allowed.filter((specifier) => refusal(file, specifier) !== null)).toEqual([])
  })
})

describe("PRD-261 A-14c — Play's conversations import the core only through the SDK", () => {
  it("import only '@automatos/theme-sdk', react, react-dom and Play's own folders", () => {
    expect(refusals()).toEqual([])
  })
})

describe("PRD-261 A-14c — Play's copies of the core stay the core's", () => {
  it('names an untitled Deliverable and the three "Made by" sources in the core\'s words', () => {
    expect(UNTITLED_DELIVERABLE).toBe(UNTITLED)
    for (const [source, word] of Object.entries(MADE_ELSEWHERE_WORDS)) {
      expect(word, source).toBe(MADE_BY_SOURCE[source])
    }
  })

  it("folds every character into one line exactly as the core's lib/text/plain-line.ts does", () => {
    const differ = comparedCharacters().filter((char) => {
      const text = `a${char}b`
      return cutDraft(text) !== oneLine(text)
    })
    expect(differ.map((char) => char.codePointAt(0)?.toString(16))).toEqual([])
  })
})
