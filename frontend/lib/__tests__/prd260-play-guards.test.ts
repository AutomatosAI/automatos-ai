// @vitest-environment node
/**
 * PRD-260 K1, K2 — the Play guards.
 *
 * (a) Play is ported, never imported: no module specifier in the frontend
 *     reaches the read-only Play reference or the automatos-play repo. Read
 *     through the TypeScript compiler API, import syntax only, so a comment
 *     or a string that mentions those paths never counts (14 Oct: a docstring
 *     tripped PRD-259's text-grep guard).
 * (b) Play's CSS defines and reads only --pl-* variables, plus the core's font
 *     and radius ones: Play's --accent and --ring would clobber Studio's HSL
 *     tokens of the same names.
 * (c) Every visible colour comes from the look: no #hex and no rgb()/hsl()
 *     literal in Play's CSS or components. lib/looks/catalogue.ts is the one
 *     home for a literal colour.
 */
import fs from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

const FRONTEND = path.resolve(__dirname, '../..')
const SOURCE_DIRS = ['app', 'components', 'lib', 'hooks', 'contexts', 'stores']
/** Build output (.next, out, coverage) sits at the frontend root, outside the scanned folders. */
const SKIP_DIRS = new Set(['node_modules'])
const SOURCE_FILE = /\.(ts|tsx)$/
const TEST_FILE = /(^|\/)__tests__\/|\.test\.tsx?$/
/** Built by parts so no line of this file reads as an import of them to a text scanner. */
const FORBIDDEN_PARTS = ['prd260-spec', 'play-ref', ['automatos', 'play'].join('-'), ['scripts', 'ralph'].join('/')]
const MOCKING_CALLS = new Set(['mock', 'doMock', 'importActual', 'importMock'])

const PLAY_COMPONENT_DIRS = ['components/play', 'components/looks', 'components/mascot']
const PLAY_THEME_CSS = 'app/play-theme.css'
const CORE_VARS_PLAY_MAY_READ = new Set(['--font-geist-sans', '--font-geist-mono', '--font-newsreader', '--radius'])
const PLAY_VAR = /^--pl-[\w-]+$/
const CATALOGUE = 'lib/looks/catalogue.ts'
const CATALOGUE_MIN_COLOURS = 20

/** #rgb, #rgba, #rrggbb, #rrggbbaa; not an HTML entity such as &#39;. */
const HEX_COLOUR = /(?<!&)#(?:[0-9a-f]{8}|[0-9a-f]{6}|[0-9a-f]{3,4})\b/gi
/** rgb(), rgba(), hsl(), hsla() with literal numbers; hsl(var(--x)) is not a literal. */
const COLOUR_FUNCTION = /\b(?:rgba?|hsla?)\(\s*[\d.]+/gi

/** Parsing ~1,200 files takes a few seconds; the default 5 s is too tight on a busy runner. */
const SCAN_TIMEOUT_MS = 120_000

function walk(dir: string, keep: (name: string) => boolean): string[] {
  if (!fs.existsSync(dir)) return []
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return SKIP_DIRS.has(entry.name) ? [] : walk(full, keep)
    return keep(entry.name) ? [full] : []
  })
}

function relative(file: string): string {
  return path.relative(FRONTEND, file).split(path.sep).join('/')
}

// --- (a) module specifiers -------------------------------------------------------------

function stringArgument(node: ts.Node | undefined): string | null {
  return node !== undefined && ts.isStringLiteralLike(node) ? node.text : null
}

function isLoaderCall(callee: ts.Expression): boolean {
  if (callee.kind === ts.SyntaxKind.ImportKeyword) return true
  if (ts.isIdentifier(callee)) return callee.text === 'require'
  return ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) &&
    callee.expression.text === 'vi' && MOCKING_CALLS.has(callee.name.text)
}

function specifierOf(node: ts.Node): string | null {
  if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) return stringArgument(node.moduleSpecifier)
  if (ts.isImportEqualsDeclaration(node) && ts.isExternalModuleReference(node.moduleReference)) {
    return stringArgument(node.moduleReference.expression)
  }
  if (ts.isCallExpression(node) && isLoaderCall(node.expression)) return stringArgument(node.arguments[0])
  if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) return stringArgument(node.argument.literal)
  return null
}

/** Every module specifier in the source: import/export declarations, import(), require(), vi.mock(), import types. */
function moduleSpecifiers(fileName: string, source: string): string[] {
  const kind = fileName.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, false, kind)
  const found: string[] = []
  const visit = (node: ts.Node): void => {
    const specifier = specifierOf(node)
    if (specifier !== null) found.push(specifier)
    ts.forEachChild(node, visit)
  }
  visit(file)
  return found
}

function forbiddenIn(fileName: string, source: string): string[] {
  return moduleSpecifiers(fileName, source).filter((specifier) => FORBIDDEN_PARTS.some((part) => specifier.includes(part)))
}

// --- (b) and (c) Play's stylesheets and components ---------------------------------------

function playCssFiles(): string[] {
  const theme = path.join(FRONTEND, PLAY_THEME_CSS)
  const modules = PLAY_COMPONENT_DIRS.flatMap((dir) => walk(path.join(FRONTEND, dir), (name) => name.endsWith('.css')))
  return fs.existsSync(theme) ? [theme, ...modules] : modules
}

function playComponentFiles(): string[] {
  return PLAY_COMPONENT_DIRS.flatMap((dir) => walk(path.join(FRONTEND, dir), (name) => SOURCE_FILE.test(name)))
    .filter((file) => !TEST_FILE.test(relative(file)))
}

function withoutCssComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, ' ')
}

/** Every custom property the stylesheet defines (`--x:`) or reads (`var(--x`). */
function customProperties(css: string): string[] {
  const bare = withoutCssComments(css)
  const defined = Array.from(bare.matchAll(/(?<![\w-])(--[\w-]+)\s*:/g), (match) => match[1] ?? '')
  const read = Array.from(bare.matchAll(/var\(\s*(--[\w-]+)/g), (match) => match[1] ?? '')
  return [...defined, ...read]
}

function isPlayVariable(name: string): boolean {
  return PLAY_VAR.test(name) || CORE_VARS_PLAY_MAY_READ.has(name)
}

function literalColours(text: string): string[] {
  return [...Array.from(text.matchAll(HEX_COLOUR), (match) => match[0]), ...Array.from(text.matchAll(COLOUR_FUNCTION), (match) => match[0])]
}

describe('PRD-260 (a) — Play is ported, never imported', () => {
  const [SPEC, REF, PLAY_REPO, RALPH] = FORBIDDEN_PARTS

  it('reads import syntax only: comments and strings that mention the reference never count', () => {
    const source = [
      `// import { lookToCssVars } from '../${RALPH}/${SPEC}/${REF}/lib/looks/css'`,
      `/** Ported from ${PLAY_REPO}: ${REF}/lib/looks/catalogue.ts */`,
      `const note = 'see ${RALPH}/${SPEC}/${REF}/README.md'`,
      "import { EMBER } from '@/lib/looks/presets'",
    ].join('\n')
    expect(forbiddenIn('sample.ts', source)).toEqual([])
    expect(moduleSpecifiers('sample.ts', source)).toEqual(['@/lib/looks/presets'])
  })

  it('catches every way of naming a module', () => {
    const source = [
      `import x from '../../${RALPH}/${SPEC}/${REF}/lib/looks/css'`,
      `export * from '${PLAY_REPO}/lib/looks'`,
      `const y = require('../${REF}/lib/mascot/reducer')`,
      `const z = () => import('../${SPEC}/play-fonts')`,
      `type T = import('../${REF}/lib/looks/types').Look`,
      `vi.mock('../${REF}/lib/client')`,
      `import w = require('../${SPEC}/w')`,
      `const view = <div>{x}</div>`,
    ].join('\n')
    expect(forbiddenIn('sample.tsx', source)).toHaveLength(7)
  })

  it('finds no import of the Play reference anywhere in the frontend', () => {
    const files = SOURCE_DIRS.flatMap((dir) => walk(path.join(FRONTEND, dir), (name) => SOURCE_FILE.test(name)))
    expect(files.length).toBeGreaterThan(0)
    const hits = files.flatMap((file) =>
      forbiddenIn(file, fs.readFileSync(file, 'utf8')).map((specifier) => `${relative(file)}: ${specifier}`),
    )
    expect(hits).toEqual([])
  }, SCAN_TIMEOUT_MS)
})

describe("PRD-260 (b) — Play's CSS uses only --pl-* variables (K2)", () => {
  it('reads definitions and var() reads, comments ignored', () => {
    const css = '/* --accent: red */ .a { --accent: red; color: var(--ring); --pl-x: 1; font: var( --font-geist-sans ); } .b--c:hover {}'
    expect(customProperties(css)).toEqual(['--accent', '--pl-x', '--ring', '--font-geist-sans'])
    expect(customProperties(css).filter((name) => !isPlayVariable(name))).toEqual(['--accent', '--ring'])
  })

  it("allows the core's font and radius variables and nothing else outside --pl-*", () => {
    for (const name of ['--pl-accent', '--pl-ground-bg', '--font-geist-sans', '--font-geist-mono', '--font-newsreader', '--radius']) {
      expect(isPlayVariable(name)).toBe(true)
    }
    for (const name of ['--accent', '--ring', '--background', '--primary', '--pl', '--font-instrument']) {
      expect(isPlayVariable(name)).toBe(false)
    }
  })

  it("finds no other variable in Play's stylesheets", () => {
    const bad = playCssFiles().flatMap((file) =>
      customProperties(fs.readFileSync(file, 'utf8'))
        .filter((name) => !isPlayVariable(name))
        .map((name) => `${relative(file)}: ${name}`),
    )
    expect(bad).toEqual([])
  })
})

describe('PRD-260 (c) — no literal colours outside the look catalogue', () => {
  it('finds the patterns it is looking for', () => {
    expect(literalColours('color: #FF6A1F; border: 1px solid #fff; fill: #1a0a0280')).toEqual(['#FF6A1F', '#fff', '#1a0a0280'])
    expect(literalColours('it&#39;s #section-2 and #main')).toEqual([])
    expect(literalColours('a: rgba(0,0,0,.4); b: hsl(20 8% 7%); c: rgb( 255 0 0 )')).toHaveLength(3)
    expect(literalColours('a: hsl(var(--x)); b: rgba(var(--pl-y), .5)')).toEqual([])
  })

  it("has no literal colour in Play's stylesheets or components", () => {
    const files = [...playCssFiles(), ...playComponentFiles()]
    const found = files.flatMap((file) =>
      literalColours(fs.readFileSync(file, 'utf8')).map((colour) => `${relative(file)}: ${colour}`),
    )
    expect(found).toEqual([])
  })

  it('keeps the catalogue as the place the colours live', () => {
    const catalogue = fs.readFileSync(path.join(FRONTEND, CATALOGUE), 'utf8')
    expect(literalColours(catalogue).length).toBeGreaterThan(CATALOGUE_MIN_COLOURS)
  })
})
