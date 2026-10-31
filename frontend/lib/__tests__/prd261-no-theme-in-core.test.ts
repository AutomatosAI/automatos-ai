// @vitest-environment node
/**
 * PRD-261 A-19f2 — no theme is left in core: builtin Play left the app (A-19f1) and lives in the
 * theme kit as an installable package, so nothing in the frontend may hold it, reach for it or
 * name it, and the docs (but the historical docs/PRDS) no longer describe it (A-19f5).
 *
 * The scan reads every file of the frontend: app, components, lib, hooks, contexts, stores, styles
 * (there is no styles folder today), public, next.config.js and middleware.ts, and the rest of the
 * frontend's own files (packages, pages, types, config, scripts, the root's), never node_modules,
 * build output or the lockfile. MUST_FIND names files it must read, so it can never pass on nothing.
 *
 * - No file is left under the paths A-19f1 deleted.
 * - No module specifier, next/font/local file or stylesheet @import/url()/composes resolves into
 *   them (read through the TypeScript compiler and the stylesheet's syntax, ./module-references.ts:
 *   imports, export-from, import(), require(), vi.mock() ids, import types), and none names the
 *   read-only Play reference, the theme kit's repository or the Ralph scripts (PRD-260's guard (a),
 *   M96). MUST_SEE names real references the reader must find.
 * - No text, comments included, holds a name builtin Play left: its variable prefix, Auto's old
 *   public folder as a URL path, the kit's repository name, its look interface (the constant and the
 *   'play' value), its <html> class, a builtin theme's style or its ref. The names are built by
 *   parts here, so this file's own text never reads as one. One exception, one block, held to still
 *   need it: the old-values case of lib/__tests__/ui-style.test.ts (the values builtin Play left in
 *   browsers and on the server read as nothing). No runtime file is excepted: A-20 lifted PA-8.
 * - No runtime file (every file that is not a test's) names Play's package id either (A-20): core
 *   holds only Classic, Studio and the theme host, and Play is an installed package like any other.
 * - The seams A-19a made for the core's own code stay (lib/__tests__/prd261-a19-seams.test.ts,
 *   which this replaces in the core ledger, with PRD-260's guards (a) and (e)).
 * - The docs case: docs/ outside docs/PRDS names no builtin Play, its host folder, Auto's old
 *   public folder, core's Play stylesheet or the deleted Play doc.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import { cssReferences, localFontPaths, moduleSpecifiers, parseSource, stringArgument } from './module-references'

const FRONTEND = path.resolve(__dirname, '../..')
const REPO = path.resolve(FRONTEND, '..')
/** Parsing every source file of the frontend takes a few seconds; the default 5 s is too tight on a busy runner. */
const SCAN_TIMEOUT_MS = 120_000

const SKIP_DIRS: ReadonlySet<string> = new Set(['node_modules', '.next', 'out', 'coverage', '.turbo', '.vercel'])
const SKIP_FILES: ReadonlySet<string> = new Set(['package-lock.json'])
/** Local environment files: never read (they may hold secrets, and CI has none). */
const ENV_FILE = /^\.env/
const LOCAL_FONT_MODULE = 'next/font/local'
const BINARY = /\.(png|jpe?g|gif|ico|webp|avif|bmp|webm|mp4|mov|woff2?|ttf|otf|eot|pdf|zip|gz)$/i
const SCRIPT = /\.(tsx?|jsx?|mjs|cjs)$/
const STYLESHEET = /\.css$/
const MODULE_EXTENSION = /\.(tsx?|jsx?|mjs|cjs|css|json)$/

/** What A-19f1 deleted, relative to the frontend. */
const DELETED_DIRS: readonly string[] = [
  'components/play', 'components/looks', 'components/mascot', 'lib/play', 'lib/looks', 'lib/mascot',
  'lib/theme-host/builtin', 'app/conversations', 'app/fonts/play', 'public/auto',
]
const DELETED_HOOKS: readonly string[] = [
  'auto-mascot', 'backend-mascot', 'celebrate-new', 'mascot-player', 'play-conversation', 'play-turn', 'shelf-mascot',
  'idle-timeout', 'recheck-countdown',
]
const DELETED_FILES: readonly string[] = ['app/play-theme.css', 'app/play-fonts.ts', ...DELETED_HOOKS.map((name) => `hooks/use-${name}.ts`)]
/** PRD-260's guard (a): the read-only Play reference, the kit's repository, the Ralph scripts. Built by parts. */
const FORBIDDEN_PARTS: readonly string[] = ['prd260-spec', 'play-ref', ['automatos', 'play'].join('-'), ['scripts', 'ralph'].join('/')]

const PLAY = ['pl', 'ay'].join('')
/** Names builtin Play left, each with what it is. Built by parts (see the header). */
const BANNED_TEXT: ReadonlyArray<readonly [string, RegExp]> = [
  ["a theme's variable prefix", new RegExp(['--', 'pl-'].join(''))],
  ["Auto's old public folder as a URL path", new RegExp(`(^|[^\\w.-])/${['au', 'to'].join('')}/`)],
  ["the theme kit's repository name", new RegExp(['automatos', PLAY].join('-'))],
  ["builtin Play's look interface constant", new RegExp(['LOOK', 'INTERFACE'].join('_'))],
  ["builtin Play's look interface", new RegExp(`\\b(interface|iface)['"]?\\s*:\\s*['"\`]${PLAY}['"\`]`)],
  ["builtin Play's <html> class", new RegExp(`html\\.${PLAY}\\b`)],
  ["a builtin theme's style", new RegExp(['theme', 'builtin', ''].join(':'))],
  ["builtin Play's ref", new RegExp(`builtin:${PLAY}\\b`)],
]

interface TextException {
  readonly file: string
  readonly why: string
  /** The nodes whose text (leading comments included) the scan skips. */
  readonly nodes: (source: ts.SourceFile) => readonly ts.Node[]
}

/** The calls `describe(<title>, …)` in a test file. */
function describeBlocks(source: ts.SourceFile, title: string): ts.Node[] {
  const found: ts.Node[] = []
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'describe' &&
      stringArgument(node.arguments[0]) === title) found.push(node)
    ts.forEachChild(node, visit)
  }
  visit(source)
  return found
}

const OLD_VALUES_CASE = 'the values builtin Play left behind name nothing'
/** Test files: the old-cookie case, the one place allowed to spell the values builtin Play left behind. */
const TEST_EXCEPTIONS: readonly TextException[] = [
  {
    file: 'lib/__tests__/ui-style.test.ts',
    why: "the old-cookie case: builtin Play's style cookie, server choice and tone cookie read as nothing",
    nodes: (source) => describeBlocks(source, OLD_VALUES_CASE),
  },
]
/** Runtime files: none (A-20 lifted PA-8, so no runtime file needs to name what builtin Play left). */
const RUNTIME_EXCEPTIONS: readonly TextException[] = []
const EXCEPTIONS: readonly TextException[] = [...TEST_EXCEPTIONS, ...RUNTIME_EXCEPTIONS]

/** A test's file: under __tests__ or __mocks__, or named *.test.* or *.spec.*. Every other file is runtime code. */
const TEST_FILE = /(^|\/)__(tests|mocks)__\/|\.(test|spec)\.[cm]?[jt]sx?$/
/** Play's package id. Built by parts (see the header). */
const PLAY_PACKAGE_ID = ['automatos', PLAY].join('/')
/** What runtime code alone may not name (A-20): Play owns nothing in core. Tests may, to install Play's package. */
const RUNTIME_BANNED_TEXT: ReadonlyArray<readonly [string, RegExp]> = [
  ["Play's package id", new RegExp(`\\b${PLAY_PACKAGE_ID}(?![\\w-])`)],
]

function isTestFile(file: string): boolean {
  return TEST_FILE.test(file)
}

/** Files the scan must read: one in each area the story names, and the seam A-19f1 edited. */
const MUST_FIND: readonly string[] = [
  'app/layout.tsx', 'app/looks/page.tsx', 'app/looks/style/page.tsx', 'components/providers.tsx',
  'components/theme-host/no-look-editor-note.tsx', 'components/theme-host/theme-look-runtime.tsx', 'lib/ui-style.ts',
  'lib/theme-host/registry.ts', 'lib/theme-host/boot.ts', 'hooks/use-studio-theme.ts', 'contexts/ui-style-context.tsx',
  'stores/chat-session-store.ts', 'public/modal-fix.js', 'next.config.js', 'middleware.ts',
]
/** Real references the reader must find: an import and a next/font/local file. */
const MUST_SEE: readonly string[] = [
  'app/looks/page.tsx: @/components/theme-host/no-look-editor-note',
  'app/core-fonts.ts: ./fonts/core/geist-latin-wght-normal.woff2',
]

function filesUnder(rel: string): string[] {
  return readdirSync(path.join(FRONTEND, rel)).flatMap((entry) => {
    const child = rel === '' ? entry : `${rel}/${entry}`
    if (statSync(path.join(FRONTEND, child)).isDirectory()) {
      return SKIP_DIRS.has(entry) || entry.startsWith('.') ? [] : filesUnder(child)
    }
    return SKIP_FILES.has(entry) || ENV_FILE.test(entry) || BINARY.test(entry) ? [] : [child]
  })
}

function read(file: string): string {
  return readFileSync(path.join(FRONTEND, file), 'utf8')
}

/** The frontend path a reference written in `file` names; null for a package or a URL. */
function targetOf(file: string, reference: string): string | null {
  if (reference.startsWith('@/')) return reference.slice(2)
  if (reference.startsWith('.')) return path.posix.normalize(path.posix.join(path.posix.dirname(file), reference))
  if (reference.startsWith('/') && !reference.startsWith('//') && STYLESHEET.test(file)) return `public${reference}`
  return null
}

function isDeleted(target: string): boolean {
  const bare = target.split(/[?#]/)[0]
  if (DELETED_DIRS.some((dir) => bare === dir || bare.startsWith(`${dir}/`))) return true
  return DELETED_FILES.some((file) => bare === file || bare.replace(MODULE_EXTENSION, '') === file.replace(MODULE_EXTENSION, ''))
}

/** Every module, font file and stylesheet reference in `file`. */
function referencesOf(file: string, text: string): string[] {
  if (STYLESHEET.test(file)) return cssReferences(text)
  if (!SCRIPT.test(file)) return []
  const fonts = text.includes(LOCAL_FONT_MODULE) ? localFontPaths(file, text) : []
  return [...moduleSpecifiers(file, text), ...fonts]
}

function refusedReference(file: string, reference: string): boolean {
  const target = targetOf(file, reference)
  return (target !== null && isDeleted(target)) || FORBIDDEN_PARTS.some((part) => reference.includes(part))
}

/** `text` with the excepted nodes blanked out (newlines kept, so line numbers hold). */
function withoutExceptions(file: string, text: string): string {
  const exception = EXCEPTIONS.find((each) => each.file === file)
  if (exception === undefined) return text
  const source = parseSource(file, text)
  return exception.nodes(source).reduce((kept, node) => {
    const start = node.getFullStart()
    const blank = kept.slice(start, node.getEnd()).replace(/[^\n]/g, ' ')
    return kept.slice(0, start) + blank + kept.slice(node.getEnd())
  }, text)
}

/** `file:line: what` for every banned name in `text`. */
function bannedIn(file: string, text: string, banned: ReadonlyArray<readonly [string, RegExp]>): string[] {
  return text.split('\n').flatMap((line, index) =>
    banned.filter(([, pattern]) => pattern.test(line)).map(([what]) => `${file}:${index + 1}: ${what}`))
}

const FILES = filesUnder('')
const TEXTS: ReadonlyMap<string, string> = new Map(FILES.map((file) => [file, read(file)]))
const parsed: { references: readonly string[] | null } = { references: null }

/** `file: reference` for every reference in the frontend; parsed once, inside the first test that asks. */
function allReferences(): readonly string[] {
  if (parsed.references === null) {
    parsed.references = FILES.flatMap((file) => referencesOf(file, TEXTS.get(file) ?? '').map((reference) => `${file}: ${reference}`))
  }
  return parsed.references
}

function refusedReferences(): string[] {
  return allReferences().filter((entry) => {
    const at = entry.indexOf(': ')
    return refusedReference(entry.slice(0, at), entry.slice(at + 2))
  })
}

describe('PRD-261 A-19f2 — the scan reads the whole frontend', () => {
  it('finds every file it must, and sees the references it must', () => {
    expect(MUST_FIND.filter((file) => !TEXTS.has(file))).toEqual([])
    expect(MUST_SEE.filter((entry) => !allReferences().includes(entry))).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it('resolves a reference to what it names, and tells a deleted path from a kept one', () => {
    const auto = ['', ['au', 'to'].join(''), 'idle.png'].join('/')
    const refused: ReadonlyArray<readonly [string, string]> = [
      ['app/layout.tsx', './play-fonts'], ['components/theme-host/x.tsx', '../play/shell/play-frame'],
      ['lib/theme-host/boot.ts', './builtin/play-boot'], ['hooks/x.ts', '@/hooks/use-play-turn'],
      ['app/core-fonts.ts', './fonts/play/a.woff2'], ['app/globals.css', './play-theme.css'], ['app/globals.css', auto],
      ['x.ts', '@/lib/looks/presets'], ['x.ts', `../${FORBIDDEN_PARTS[1]}/lib/client`],
    ]
    const kept: ReadonlyArray<readonly [string, string]> = [
      ['app/layout.tsx', './core-fonts'], ['x.ts', '@/components/playbooks/list'], ['x.ts', '@/lib/look-kit/scope'],
      ['x.ts', '@/hooks/use-playbook-api'], ['x.ts', 'react'], ['x.tsx', auto], ['app/globals.css', 'https://example.com/a.png'],
    ]
    expect(refused.filter(([file, reference]) => !refusedReference(file, reference))).toEqual([])
    expect(kept.filter(([file, reference]) => refusedReference(file, reference))).toEqual([])
  })

  it('catches every banned name, and no lookalike', () => {
    // Each sample with what it must be refused as; the retired style cookie is both a builtin style and Play's ref.
    const samples: ReadonlyArray<readonly [string, readonly number[]]> = [
      [`color: var(${['--', 'pl-accent'].join('')})`, [0]], [`src="/${['au', 'to'].join('')}/hello.webm"`, [1]],
      [`key: '${['automatos', PLAY].join('-')}.looks'`, [2]], [`import { ${['LOOK', 'INTERFACE'].join('_')} } from './types'`, [3]],
      [`{ iface: '${PLAY}' }`, [4]], [`:where(html.${PLAY}) body {}`, [5]], [`'${['theme', 'builtin', PLAY].join(':')}'`, [6, 7]],
      [`choice: '${['builtin', PLAY].join(':')}'`, [7]],
    ]
    expect(samples.map(([sample]) => bannedIn('x', sample, BANNED_TEXT)))
      .toEqual(samples.map(([, hits]) => hits.map((index) => `x:1: ${BANNED_TEXT[index][0]}`)))
    expect(new Set(samples.flatMap(([, hits]) => hits)).size).toBe(BANNED_TEXT.length)
    const lookalikes = ['reports/auto/pricing.md', 'display: flex', `video.${PLAY}()`, `'${PLAY}'`, `html.${PLAY}books`, 'builtin:x']
    expect(lookalikes.flatMap((sample) => bannedIn('x', sample, BANNED_TEXT))).toEqual([])
  })

  it("refuses Play's package id in runtime code, and tells a test's file from a runtime one (A-20)", () => {
    const samples = [`themeId: '${PLAY_PACKAGE_ID}'`, `/api/themes/looks/${PLAY_PACKAGE_ID}`, `(\`${PLAY_PACKAGE_ID}\`)`]
    expect(samples.map((sample) => bannedIn('x', sample, RUNTIME_BANNED_TEXT).length)).toEqual(samples.map(() => 1))
    const lookalikes = [`${PLAY_PACKAGE_ID}books`, `${PLAY_PACKAGE_ID}-lite`, 'automatos-e2e/hello', `x${PLAY_PACKAGE_ID}`]
    expect(lookalikes.flatMap((sample) => bannedIn('x', sample, RUNTIME_BANNED_TEXT))).toEqual([])
    const tests = [
      'lib/__tests__/ui-style.test.ts', 'lib/theme-host/__tests__/installed-entry-fixture.ts',
      'components/widgets/CodingCanvasWidget/DiffCard.test.tsx', 'lib/__mocks__/x.ts', 'x.spec.mjs',
    ]
    const runtime = ['lib/theme-host/package-manifest.ts', 'lib/api-client.ts', 'lib/testing.ts', 'app/tests/page.tsx', 'lib/x.test-utils.ts']
    expect(tests.filter((file) => !isTestFile(file))).toEqual([])
    expect(runtime.filter(isTestFile)).toEqual([])
  })
})

describe('PRD-261 A-19f2 — builtin Play is gone from the frontend', () => {
  it('leaves no file under the paths A-19f1 deleted', () => {
    expect([...DELETED_DIRS, ...DELETED_FILES].filter((rel) => existsSync(path.join(FRONTEND, rel)))).toEqual([])
    expect(FILES.filter(isDeleted)).toEqual([])
  })

  it('has no reference into them, nor to the Play reference, the kit repository or the Ralph scripts', () => {
    expect(refusedReferences()).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it('names nothing builtin Play left, in code, strings or comments, outside the one exception', () => {
    const found = FILES.flatMap((file) => bannedIn(file, withoutExceptions(file, TEXTS.get(file) ?? ''), BANNED_TEXT))
    expect(found).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it("names Play's package id in no runtime file, and excepts no runtime file (A-20)", () => {
    const runtime = FILES.filter((file) => !isTestFile(file))
    expect(runtime).toContain('lib/theme-host/package-manifest.ts')
    expect(runtime.flatMap((file) => bannedIn(file, TEXTS.get(file) ?? '', RUNTIME_BANNED_TEXT))).toEqual([])
    expect(RUNTIME_EXCEPTIONS).toEqual([])
    expect(EXCEPTIONS.filter(({ file }) => !isTestFile(file)).map(({ file }) => file)).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it('keeps each exception to one block that still needs it', () => {
    const stale = EXCEPTIONS.filter(({ file, nodes }) => {
      const text = TEXTS.get(file) ?? ''
      const blocks = nodes(parseSource(file, text))
      return blocks.length !== 1 || blocks.every((node) => bannedIn(file, text.slice(node.getFullStart(), node.getEnd()), BANNED_TEXT).length === 0)
    })
    expect(stale.map(({ file, why }) => `${file} (${why})`)).toEqual([])
  })
})

/** Each importer A-19a switched to a core seam, and the seam it names (prd261-a19-seams.test.ts). */
const SWITCHED: ReadonlyArray<readonly [string, string]> = [
  ['lib/chat/message-text.ts', '@/lib/chat/chat-status'],
  ['hooks/use-refresh-history-on-turn-end.ts', '@/lib/chat/chat-status'],
  ['components/theme-host/theme-slot-page.tsx', '@/lib/theme-host/focus'],
  ['lib/theme-sdk/chat.ts', '@/stores/fresh-draft-store'],
  ['hooks/use-start-chat.ts', '@/stores/fresh-draft-store'],
  ['components/theme-host/account-look-scope.tsx', '@/lib/look-kit/scope'],
  ['components/theme-host/look-scope.ts', '@/lib/look-kit/scope'],
]

describe("PRD-261 A-19a — the core's own seams stay the core's", () => {
  it('each importer A-19a switched still names its core seam', () => {
    const missing = SWITCHED.filter(([file, seam]) => !allReferences().includes(`${file}: ${seam}`))
    expect(missing.map(([file, seam]) => `${file}: ${seam}`)).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it('keeps the moved fresh-draft store and look scope, with no copy of the store at its old path', () => {
    expect(existsSync(path.join(FRONTEND, 'stores/play-chat-store.ts'))).toBe(false)
    expect(existsSync(path.join(FRONTEND, 'stores/fresh-draft-store.ts'))).toBe(true)
    expect(existsSync(path.join(FRONTEND, 'lib/look-kit/scope.ts'))).toBe(true)
  })

  it('keeps hooks/use-media-preferences.ts in core, read by the SDK context', () => {
    expect(allReferences()).toContain('lib/theme-sdk/context.ts: @/hooks/use-media-preferences')
  }, SCAN_TIMEOUT_MS)
})

const DOCS = 'docs'
/** Historical: the PRDs (and the run checklists kept with them) describe the app as it was. */
const HISTORICAL_DOCS = 'docs/PRDS'
const DOC_FILE = /\.(md|txt)$/
const DOC_BANNED: ReadonlyArray<readonly [string, RegExp]> = [
  ['builtin Play', /\bbuiltin[ -]play\b/i],
  ["builtin Play's host folder", /lib\/theme-host\/builtin\b/],
  ["Auto's old public folder", /public\/auto\b/],
  ["core's Play stylesheet (the kit's own path is fine)", /(?<!src\/)app\/play-theme\.css/],
  ['the deleted Play doc', /frontend-architecture\/play-theme\.md/],
]
/** Docs the case must read: the surface map and the two A-19f5 rewrote. */
const DOCS_MUST_FIND: readonly string[] = [
  'docs/theme-host/sdk-surface-map.md', 'docs/frontend-architecture/_index.md', 'docs/theme-host/css-layer-model.md',
]

function docsUnder(rel: string): string[] {
  return readdirSync(path.join(REPO, rel)).flatMap((entry) => {
    const child = `${rel}/${entry}`
    if (statSync(path.join(REPO, child)).isDirectory()) return child === HISTORICAL_DOCS ? [] : docsUnder(child)
    return DOC_FILE.test(entry) ? [child] : []
  })
}

describe('PRD-261 A-19f5 — the docs no longer describe builtin Play', () => {
  const docs = docsUnder(DOCS)

  it('reads every doc but the historical ones', () => {
    expect(DOCS_MUST_FIND.filter((file) => !docs.includes(file))).toEqual([])
    expect(docs.filter((file) => file.startsWith(`${HISTORICAL_DOCS}/`))).toEqual([])
  })

  it('catches each name, and leaves the kit path of Play\'s stylesheet alone', () => {
    const samples = ['Builtin Play keeps', 'lib/theme-host/builtin/x.ts', 'public/auto/idle.png', '`frontend/app/play-theme.css`', 'docs/frontend-architecture/play-theme.md']
    expect(samples.map((sample) => bannedIn('x', sample, DOC_BANNED).length)).toEqual(samples.map(() => 1))
    expect(bannedIn('x', 'themes/play/src/app/play-theme.css', DOC_BANNED)).toEqual([])
  })

  it('names none of them', () => {
    expect(docs.flatMap((file) => bannedIn(file, readFileSync(path.join(REPO, file), 'utf8'), DOC_BANNED))).toEqual([])
  })
})
