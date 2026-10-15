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
 * (d) M01, M16: Play's own type, inline code and sheet colours win over
 *     Studio's element rules, which reach Play's elements because <html>
 *     carries studio and play together. Proven statically (css: false).
 * (e) M17: the core's seams (the chrome, every layout, the 404) load none of
 *     Play's screens with them: the static import graph from a seam reaches
 *     components/play only at the switches and the lazy slots, whose screen
 *     comes through import() when it renders.
 */
import fs from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import {
  compareSpecificity,
  cssRules,
  declaredProperties,
  elementOf,
  lastCompound,
  namesClass,
  specificity,
  splitTopLevel,
} from './css-cascade'

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

// --- (d) the cascade on Play's elements ---------------------------------------------------

const GLOBALS_CSS = 'app/globals.css'
const SHEET_CSS = 'components/play/deliverables/deliverable-content.module.css'
const PLAY_PREFIX = ':global(html.play)'
/** The elements Studio's element rules restyle: headings, keyboard keys and inline code. */
const GUARDED_TAGS = new Set(['h1', 'h2', 'kbd', 'code'])
const CHIP_TAGS = new Set(['kbd', 'code'])
/** A code block's code: Studio's chip rule (code:not(pre code)) never reaches it. */
const INSIDE_PRE = /(?:^|[\s>+~])pre(?![\w-])/
const MARKDOWN_RENDERER = 'MarkdownView'
/** 7 h1, 16 h2, 1 kbd and 2 code carried a module class on 15 Oct (the review fix plan, P260-FX-01). */
const GUARDED_ELEMENTS_MIN = 26
/** The Deliverable sheet and Auto's reply. */
const MARKDOWN_SURFACES_MIN = 2
const SHEET_FOCUS = /--pl-focus-outline\s*:[^;]*var\(\s*--pl-color-sheet-accent\s*\)/
/** Studio's chip rule sets all four, so Play's inline kbd and code rules set each (either name). */
const CHIP_PROPERTIES = [['background', 'background-color'], ['padding'], ['border-radius'], ['font', 'font-size']]
/** Studio's element rules in globals.css: they reach every Play element, since <html> carries studio. */
const STUDIO_ELEMENT_RULES: Readonly<Record<string, readonly string[]>> = {
  h1: ['.studio h1'],
  h2: ['.studio h2'],
  kbd: ['.studio kbd'],
  code: ['.studio code', '.studio code:not(pre code)'],
}
/** The core markdown's rules in globals.css: they reach the elements inside a MarkdownView. */
const MARKDOWN_ELEMENT_RULES: Readonly<Record<string, readonly string[]>> = {
  h1: ['.md-view h1', '.studio .md-view h1'],
  h2: ['.md-view h2', '.studio .md-view h2'],
  h3: ['.md-view h3', '.studio .md-view h3'],
  code: ['.md-view code', '.md-view pre code'],
  a: ['.md-view a', '.studio .md-view a'],
  strong: ['.md-view strong'],
}

interface StyledElement {
  /** file:line of the JSX element. */
  readonly where: string
  readonly tag: string
  /** The CSS module the class comes from (absolute path). */
  readonly css: string
  readonly className: string
}

interface PlayRule {
  readonly css: string
  readonly selector: string
  /** The element the rule lands on: the JSX tag, or the selector's own element name. */
  readonly tag: string
  readonly body: string
  /** A rule of a markdown surface, where the core markdown's rules reach too. */
  readonly inMarkdown: boolean
}

function resolveSpecifier(fromFile: string, specifier: string): string {
  return specifier.startsWith('@/') ? path.join(FRONTEND, specifier.slice(2)) : path.resolve(path.dirname(fromFile), specifier)
}

/** `import styles from './x.module.css'` → styles → the module's path. */
function cssModuleBindings(file: ts.SourceFile): Map<string, string> {
  const bindings = new Map<string, string>()
  for (const statement of file.statements) {
    if (!ts.isImportDeclaration(statement)) continue
    const local = statement.importClause?.name?.text
    const specifier = stringArgument(statement.moduleSpecifier)
    if (local && specifier?.endsWith('.module.css')) bindings.set(local, resolveSpecifier(file.fileName, specifier))
  }
  return bindings
}

/** The module classes a className names: styles.x, also inside cn() or a template literal. */
function moduleClasses(value: ts.Node, bindings: Map<string, string>): Array<{ css: string; className: string }> {
  const found: Array<{ css: string; className: string }> = []
  const visit = (node: ts.Node): void => {
    if (ts.isPropertyAccessExpression(node) && ts.isIdentifier(node.expression)) {
      const css = bindings.get(node.expression.text)
      if (css !== undefined) found.push({ css, className: node.name.text })
    }
    ts.forEachChild(node, visit)
  }
  visit(value)
  return found
}

function elementClasses(file: ts.SourceFile, element: ts.JsxOpeningLikeElement, bindings: Map<string, string>): StyledElement[] {
  const attribute = element.attributes.properties.find(
    (property) => ts.isJsxAttribute(property) && ts.isIdentifier(property.name) && property.name.text === 'className',
  )
  const value = attribute !== undefined && ts.isJsxAttribute(attribute) ? attribute.initializer : undefined
  if (value === undefined) return []
  const tag = ts.isIdentifier(element.tagName) ? element.tagName.text : ''
  const where = `${relative(file.fileName)}:${file.getLineAndCharacterOfPosition(element.getStart(file)).line + 1}`
  return moduleClasses(value, bindings).map((named) => ({ where, tag, ...named }))
}

/** Every JSX element in the file whose className names a CSS-module class. */
function styledElements(fileName: string, source: string): StyledElement[] {
  const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  const bindings = cssModuleBindings(file)
  const found: StyledElement[] = []
  const visit = (node: ts.Node): void => {
    if (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) found.push(...elementClasses(file, node, bindings))
    ts.forEachChild(node, visit)
  }
  visit(file)
  return found
}

function allStyledElements(): StyledElement[] {
  return playComponentFiles()
    .filter((file) => file.endsWith('.tsx'))
    .flatMap((file) => styledElements(file, fs.readFileSync(file, 'utf8')))
}

function rulesOf(css: string): Array<{ selector: string; body: string }> {
  return cssRules(fs.readFileSync(css, 'utf8')).flatMap((rule) => rule.selectors.map((selector) => ({ selector, body: rule.body })))
}

/** The rules that style the element's class: those whose last compound names it. */
function rulesForElement(element: StyledElement): PlayRule[] {
  return rulesOf(element.css)
    .filter(({ selector }) => namesClass(lastCompound(selector), element.className))
    .map(({ selector, body }) => ({ css: element.css, selector, tag: element.tag, body, inMarkdown: false }))
}

/** Every rule of a markdown surface: any selector naming the class on the MarkdownView. */
function rulesForMarkdown(surface: StyledElement): PlayRule[] {
  return rulesOf(surface.css)
    .filter(({ selector }) => namesClass(selector, surface.className))
    .map(({ selector, body }) => ({ css: surface.css, selector, tag: elementOf(lastCompound(selector)), body, inMarkdown: true }))
}

function guardedRules(): PlayRule[] {
  const elements = allStyledElements()
  return [
    ...elements.filter((element) => GUARDED_TAGS.has(element.tag)).flatMap(rulesForElement),
    ...elements.filter((element) => element.tag === MARKDOWN_RENDERER).flatMap(rulesForMarkdown),
  ]
}

function label(rule: PlayRule): string {
  return `${relative(rule.css)}: ${rule.selector}`
}

/** One line per finding: a class on several elements (edges' .title) is one rule. */
function unique(findings: readonly string[]): string[] {
  return Array.from(new Set(findings))
}

/** The core rules the Play rule ties with or loses to, on the element it lands on. */
function outrankedBy(rule: PlayRule): string[] {
  const own = specificity(rule.selector)
  const markdown = rule.inMarkdown ? (MARKDOWN_ELEMENT_RULES[rule.tag] ?? []) : []
  return [...(STUDIO_ELEMENT_RULES[rule.tag] ?? []), ...markdown].filter((core) => compareSpecificity(own, specificity(core)) <= 0)
}

/** Inline kbd and code: the chip properties none of the rules for that target sets. */
function missingChipProperties(rules: readonly PlayRule[]): string[] {
  const declared = new Map<string, Set<string>>()
  for (const rule of rules.filter((each) => CHIP_TAGS.has(each.tag) && !INSIDE_PRE.test(each.selector))) {
    const key = `${relative(rule.css)}: ${lastCompound(rule.selector)}`
    declared.set(key, new Set([...(declared.get(key) ?? []), ...declaredProperties(rule.body)]))
  }
  return Array.from(declared).flatMap(([key, names]) =>
    CHIP_PROPERTIES.filter((either) => !either.some((name) => names.has(name))).map((either) => `${key} sets no ${either.join('/')}`),
  )
}

describe("PRD-260 (d) — Play's own type, inline code and sheet colours win over Studio's element rules (M01, M16)", () => {
  it('reads rules, selector lists and specificity', () => {
    const css = '@tailwind base; /* .x { } */ .a, .b:is(.c, .d) { color: red } @media (min-width: 1px) { .e h1 { margin: 0 } } @keyframes k { from { opacity: 0 } }'
    expect(cssRules(css).map((rule) => rule.selectors)).toEqual([['.a', '.b:is(.c, .d)'], ['.e h1']])
    expect(splitTopLevel(':global(html.play) .x,\n  .y  z')).toEqual([':global(html.play) .x', '.y z'])
    expect(lastCompound(':global(html.play) .prose > :first-child')).toBe(':first-child')
    expect(elementOf(lastCompound('.prose li::marker'))).toBe('li')
    expect(namesClass('.titleRow', 'title')).toBe(false)
    expect(declaredProperties(' font: var(--x); --pl-y: 1px ;')).toEqual(['font', '--pl-y'])
    expect(INSIDE_PRE.test('.prose pre code')).toBe(true)
    expect(INSIDE_PRE.test('.pre-wrap code')).toBe(false)
    expect(specificity('.studio h1')).toEqual([0, 1, 1])
    expect(specificity('.studio code:not(pre code)')).toEqual([0, 1, 3])
    expect(specificity('.studio .md-view a')).toEqual([0, 2, 1])
    expect(specificity(':global(html.play) .title')).toEqual([0, 2, 1])
    expect(specificity(":global(html.play) .item[data-newest='true'] .cardTitle")).toEqual([0, 4, 1])
    expect(specificity(':where(html.play :focus-visible)')).toEqual([0, 0, 0])
    expect(specificity('.prose li::marker')).toEqual([0, 1, 2])
  })

  it('finds the core rules it compares against in globals.css', () => {
    const core = new Set(rulesOf(path.join(FRONTEND, GLOBALS_CSS)).map(({ selector }) => selector))
    const named = [...Object.values(STUDIO_ELEMENT_RULES), ...Object.values(MARKDOWN_ELEMENT_RULES)].flat()
    expect(named.filter((selector) => !core.has(selector))).toEqual([])
  })

  it("finds Play's headings, keys and inline code, and both markdown surfaces", () => {
    const elements = allStyledElements()
    expect(elements.filter((element) => GUARDED_TAGS.has(element.tag)).length).toBeGreaterThanOrEqual(GUARDED_ELEMENTS_MIN)
    expect(elements.filter((element) => element.tag === MARKDOWN_RENDERER).length).toBeGreaterThanOrEqual(MARKDOWN_SURFACES_MIN)
  })

  it('starts every rule on those elements, and every markdown-surface rule, with html.play', () => {
    expect(unique(guardedRules().filter((rule) => !rule.selector.startsWith(PLAY_PREFIX)).map(label))).toEqual([])
  })

  it('outranks every globals.css rule that reaches the same element (.studio h1, .studio code:not(pre code), .studio .md-view a)', () => {
    const lost = guardedRules().flatMap((rule) => outrankedBy(rule).map((core) => `${label(rule)} does not beat ${core}`))
    expect(unique(lost)).toEqual([])
  })

  it("sets the background, padding, radius and size of inline kbd and code itself, since Studio's chip sets all four", () => {
    expect(missingChipProperties(guardedRules())).toEqual([])
  })

  it("draws the keyboard ring on the Deliverable sheet in the sheet's accent (M16)", () => {
    const sheet = rulesOf(path.join(FRONTEND, SHEET_CSS)).find(({ selector }) => selector === '.document')
    expect(sheet?.body).toMatch(SHEET_FOCUS)
  })
})

// --- (e) the core's seams load Play's screens on demand (M17) -----------------------------

/** The chrome every page renders: MainLayout, the headers, the sidebars. */
const CHROME_DIR = 'components/layout'
/** The files every page of a route loads with it: its layouts and its 404. */
const APP_SEAM = /^(layout|not-found)\.tsx$/
const PLAY_DIR = 'components/play/'
/** A slot: a 'use client' module that fetches one Play screen with import() when it renders. */
const LAZY_SLOT = /-lazy\.tsx$/
/** The switches that pick Play's screen or the core's: they hold no Play design of their own. */
const PLAY_SWITCHES = new Set(['components/play/play-route-switch.tsx', 'components/play/edges/not-found-view.tsx'])
/** Each seam and the slot its Play screen comes through. */
const SEAM_SLOTS: ReadonlyArray<readonly [string, string]> = [
  ['components/layout/main-layout.tsx', 'components/play/shell/play-frame-lazy.tsx'],
  ['app/chat/layout.tsx', 'components/play/chat/play-chat-lazy.tsx'],
  ['app/deliverables/layout.tsx', 'components/play/deliverables/play-shelf-lazy.tsx'],
  ['app/not-found.tsx', 'components/play/edges/lost-screen-lazy.tsx'],
]
const MODULE_SUFFIXES = ['', '.tsx', '.ts', '/index.tsx', '/index.ts']

type LoadKind = 'static' | 'dynamic'

/** How a node loads a module: a declaration or require() loads it with the file, import() on demand; type-only never. */
function loadKindOf(node: ts.Node): LoadKind | null {
  if (ts.isImportDeclaration(node)) return node.importClause?.isTypeOnly ? null : 'static'
  if (ts.isExportDeclaration(node)) return node.isTypeOnly ? null : 'static'
  if (ts.isImportEqualsDeclaration(node)) return node.isTypeOnly ? null : 'static'
  if (!ts.isCallExpression(node)) return null
  if (node.expression.kind === ts.SyntaxKind.ImportKeyword) return 'dynamic'
  return ts.isIdentifier(node.expression) && node.expression.text === 'require' ? 'static' : null
}

/** The module specifiers the source loads in that way. */
function loadsOf(fileName: string, source: string, kind: LoadKind): string[] {
  const scriptKind = fileName.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, false, scriptKind)
  const found: string[] = []
  const visit = (node: ts.Node): void => {
    const specifier = loadKindOf(node) === kind ? specifierOf(node) : null
    if (specifier !== null) found.push(specifier)
    ts.forEachChild(node, visit)
  }
  visit(file)
  return found
}

/** The frontend file a local specifier names, relative to the frontend; null for a package. */
function localModule(fromFile: string, specifier: string): string | null {
  if (!specifier.startsWith('@/') && !specifier.startsWith('.')) return null
  const base = resolveSpecifier(fromFile, specifier)
  const hit = MODULE_SUFFIXES.map((suffix) => base + suffix).find((candidate) => fs.existsSync(candidate) && fs.statSync(candidate).isFile())
  return hit === undefined ? null : relative(hit)
}

/** The local modules a frontend file loads in that way (a stylesheet loads none). */
function localLoads(rel: string, kind: LoadKind): string[] {
  if (!SOURCE_FILE.test(rel)) return []
  const full = path.join(FRONTEND, rel)
  return loadsOf(full, fs.readFileSync(full, 'utf8'), kind).flatMap((specifier) => localModule(full, specifier) ?? [])
}

/** Every local module a seam loads with it, however deep, through Play's modules and the core's alike. */
function staticGraph(seam: string): string[] {
  const reached = new Set<string>([seam])
  const queue = [seam]
  for (let next = queue.pop(); next !== undefined; next = queue.pop()) {
    for (const dependency of localLoads(next, 'static')) {
      if (reached.has(dependency)) continue
      reached.add(dependency)
      queue.push(dependency)
    }
  }
  return Array.from(reached)
}

function seamFiles(): string[] {
  const chrome = walk(path.join(FRONTEND, CHROME_DIR), (name) => SOURCE_FILE.test(name))
  const app = walk(path.join(FRONTEND, 'app'), (name) => APP_SEAM.test(name))
  return [...chrome, ...app].map(relative).filter((rel) => !TEST_FILE.test(rel))
}

function isSlot(rel: string): boolean {
  return LAZY_SLOT.test(rel) || PLAY_SWITCHES.has(rel)
}

describe("PRD-260 (e) — the core's seams load Play's screens only when Play renders them (M17)", () => {
  it('reads how a module loads another: declarations and require() with it, import() on demand, types never', () => {
    const source = [
      "import { PlayFrame } from '@/components/play/shell/play-frame'",
      "export { PlayChat } from './play-chat'",
      "import './frame.module.css'",
      "const legacy = require('../edges/lost-screen')",
      "import type { PlayFrameProps } from '@/components/play/shell/play-frame'",
      "export type { Look } from '@/lib/looks/types'",
      "type Shelf = import('@/components/play/deliverables/play-shelf').PlayShelfProps",
      "const Lazy = dynamic(() => import('./play-shelf').then((m) => m.PlayShelf))",
      "// import { LostScreen } from '@/components/play/edges/lost-screen'",
      "const note = 'components/play/chat/play-chat'",
    ].join('\n')
    expect(loadsOf('sample.tsx', source, 'static')).toEqual([
      '@/components/play/shell/play-frame',
      './play-chat',
      './frame.module.css',
      '../edges/lost-screen',
    ])
    expect(loadsOf('sample.tsx', source, 'dynamic')).toEqual(['./play-shelf'])
  })

  it("finds the seams, and walks a seam's graph through the core's modules", () => {
    const seams = seamFiles()
    expect(seams).toEqual(expect.arrayContaining(['app/layout.tsx', ...SEAM_SLOTS.map(([seam]) => seam)]))
    expect(seams.some((rel) => rel.includes('__tests__'))).toBe(false)
    const graph = staticGraph('app/chat/layout.tsx')
    expect(graph).toContain('components/play/play-route-switch.tsx')
    expect(graph).toContain('components/chatbot/discussion-bar.tsx')
    expect(staticGraph('components/layout/main-layout.tsx')).toContain('components/layout/studio-header.tsx')
  })

  it('reaches components/play from a seam only at the switches and the lazy slots, however deep', () => {
    const loaded = seamFiles().flatMap((seam) =>
      staticGraph(seam)
        .filter((rel) => rel.startsWith(PLAY_DIR) && !isSlot(rel))
        .map((rel) => `${seam} loads ${rel}`),
    )
    expect(loaded).toEqual([])
  }, SCAN_TIMEOUT_MS)

  it('mounts each Play screen through its slot, which fetches it with import() and loads nothing of Play with it', () => {
    for (const [seam, slot] of SEAM_SLOTS) {
      expect(staticGraph(seam), seam).toContain(slot)
      expect(localLoads(slot, 'static').filter((rel) => rel.startsWith(PLAY_DIR)), slot).toEqual([])
      expect(localLoads(slot, 'dynamic').filter((rel) => rel.startsWith(PLAY_DIR) && !isSlot(rel)), slot).toHaveLength(1)
    }
  })
})

// --- (e) Make it yours on Paper, in forced colours and with a long name ---------------------
//
// M18: the raw accents fall below 3:1 on Paper (1.07:1 for cream to 2.94:1 for rose), so a
// pressed or current state is marked in the accent as text, and an accent swatch has an edge
// of its own. M19: forced colours (Windows High Contrast) drop box-shadows and repaint
// backgrounds, so the colour samples opt out, pressed edges widen, every pressed control is
// ringed in Highlight, and no focus style hides the outline behind a box-shadow ring. M73: a
// long look name wraps instead of squeezing the heading or being clipped in its card.

const EDITOR_CSS = 'components/looks/editor.module.css'
const LOOKS_CSS = 'components/looks/looks.module.css'
const STYLE_CSS = 'components/looks/style/style.module.css'
/** A rule on a pressed or current state: it marks what is chosen, or where you are. */
const STATE_SELECTOR = /\[aria-(?:pressed|current)\b/
/** The raw accent (--pl-accent, --pl-color-accent): not its text colour, ink or a tint. */
const RAW_ACCENT = /var\(\s*--pl-(?:color-)?accent\s*[,)]/
/** Play's CSS held 8 rules on a pressed or current state on 15 Oct. */
const STATE_RULES_MIN = 8
const FORCED_COLOURS_BLOCK = /@media\s*\(\s*forced-colors\s*:\s*active\s*\)\s*\{/g
/** `outline: none` or `0`, `outline-style: none`, `outline-width: 0`: no ring forced colours could draw. */
const HIDDEN_OUTLINE: Readonly<Record<string, RegExp>> = {
  outline: /^(?:none|0(?:px)?)(?:\s*!important)?$/,
  'outline-style': /^none\b/,
  'outline-width': /^0(?:px)?\b/,
}
/** The colour samples, file by file: each keeps its own colour in forced colours. */
const COLOUR_SAMPLES: Readonly<Record<string, readonly string[]>> = {
  [EDITOR_CSS]: ['.swatch', '.groundSwatch', '.dot', '.lightDot'],
  [LOOKS_CSS]: ['.cardTop', '.cardLight', '.cardAccent'],
  [STYLE_CSS]: ['.swatchChip', '.step'],
}
/** The pressed edges forced colours widen (their ring is a box-shadow, which forced colours drop). */
const WIDENED_IN_FORCED_COLOURS: Readonly<Record<string, readonly string[]>> = {
  [EDITOR_CSS]: ['.option[aria-pressed="true"]', '.typeOption[aria-pressed="true"]', '.chip[aria-pressed="true"]'],
  [LOOKS_CSS]: ['.card[aria-pressed="true"]'],
}
const PRESSED_EDGE_MIN_PX = 2
const PRESSED_ANYWHERE = 'html.play [aria-pressed="true"]'

type Rules = ReturnType<typeof cssRules>

function readFrontend(file: string): string {
  return fs.readFileSync(path.join(FRONTEND, file), 'utf8')
}

/** A selector compared as written, whichever quotes its attribute value uses. */
function sameSelector(a: string, b: string): boolean {
  return a.replace(/'/g, '"') === b.replace(/'/g, '"')
}

/** The bodies of every rule whose selector list names exactly this selector, in source order. */
function bodyOf(rules: Rules, selector: string): string {
  return rules
    .filter((rule) => rule.selectors.some((each) => sameSelector(each, selector)))
    .map((rule) => rule.body)
    .join(';')
}

/** A body's declarations, property to value; a later one wins, as in the cascade. */
function declarations(body: string): ReadonlyMap<string, string> {
  const pairs = body.split(';').flatMap((declaration): Array<[string, string]> => {
    const colon = declaration.indexOf(':')
    if (colon < 0) return []
    return [[declaration.slice(0, colon).trim().toLowerCase(), declaration.slice(colon + 1).trim().replace(/\s+/g, ' ')]]
  })
  return new Map(pairs)
}

/** The index of the brace that closes the block opened at `open`. */
function blockEnd(css: string, open: number): number {
  let depth = 0
  for (let i = open; i < css.length; i++) {
    if (css[i] === '{') depth++
    else if (css[i] === '}' && --depth === 0) return i
  }
  return css.length
}

/** The rules inside the stylesheet's @media (forced-colors: active) blocks. */
function forcedColoursRules(css: string): Rules {
  const bare = withoutCssComments(css)
  return Array.from(bare.matchAll(FORCED_COLOURS_BLOCK)).flatMap((match) => {
    const open = (match.index ?? 0) + match[0].length - 1
    return cssRules(bare.slice(open + 1, blockEnd(bare, open)))
  })
}

function hidesOutline(body: string): boolean {
  const declared = declarations(body)
  return Object.entries(HIDDEN_OUTLINE).some(([property, hidden]) => hidden.test(declared.get(property) ?? ''))
}

function located(file: string, rule: Rules[number]): string {
  return `${relative(file)}: ${rule.selectors.join(', ')}`
}

describe('PRD-260 (e) — Make it yours on Paper, in forced colours and with a long name (M18, M19, M73)', () => {
  it('reads declarations, forced-colours blocks, hidden outlines and the raw accent', () => {
    const css = ".a { color: red } @media (forced-colors: active) { .b[aria-pressed='true'] { border-width: 2px } } @media (min-width: 1px) { .c { outline: none } }"
    expect(forcedColoursRules(css).map((rule) => rule.selectors)).toEqual([[".b[aria-pressed='true']"]])
    expect(bodyOf(forcedColoursRules(css), '.b[aria-pressed="true"]')).toContain('border-width')
    expect(declarations(' outline: 2px  solid transparent; outline-offset: 2px ').get('outline')).toBe('2px solid transparent')
    expect(['outline: none', 'outline:0', 'outline-style: none', 'outline-width: 0px'].map(hidesOutline)).toEqual([true, true, true, true])
    expect(['outline: 2px solid transparent', 'outline-offset: 0', 'outline: var(--pl-focus-outline)'].map(hidesOutline)).toEqual([false, false, false])
    expect(['var(--pl-color-accent)', 'var( --pl-accent , red)'].map((value) => RAW_ACCENT.test(value))).toEqual([true, true])
    expect(['var(--pl-color-accent-text)', 'var(--pl-accent-ink)', 'var(--pl-tint-accent-line)'].map((value) => RAW_ACCENT.test(value))).toEqual([false, false, false])
  })

  it('marks every pressed and current state in the accent as text, never the raw accent (M18)', () => {
    const states = playCssFiles().flatMap((file) =>
      cssRules(fs.readFileSync(file, 'utf8'))
        .filter((rule) => rule.selectors.some((selector) => STATE_SELECTOR.test(selector)))
        .map((rule) => ({ file, rule })),
    )
    expect(states.length).toBeGreaterThanOrEqual(STATE_RULES_MIN)
    expect(states.filter(({ rule }) => RAW_ACCENT.test(rule.body)).map(({ file, rule }) => located(file, rule))).toEqual([])
  })

  it('draws the glow slider in the accent as text, and edges every accent swatch in text2 under the chosen ring (M18)', () => {
    const rules = cssRules(readFrontend(EDITOR_CSS))
    expect(declarations(bodyOf(rules, '.slider')).get('accent-color')).toBe('var(--pl-color-accent-text)')
    expect(declarations(bodyOf(rules, '.swatch')).get('box-shadow')).toBe('inset 0 0 0 1px var(--pl-color-text-2)')
    expect(declarations(bodyOf(rules, '.swatch[aria-pressed="true"]')).get('box-shadow')).toBe(
      '0 0 0 2px var(--pl-color-text), inset 0 0 0 1px var(--pl-color-text-2)',
    )
  })

  it('keeps every colour sample in its own colours in forced colours (M19)', () => {
    const adjusted = Object.entries(COLOUR_SAMPLES).flatMap(([file, selectors]) => {
      const rules = cssRules(readFrontend(file))
      return selectors
        .filter((selector) => declarations(bodyOf(rules, selector)).get('forced-color-adjust') !== 'none')
        .map((selector) => `${file}: ${selector}`)
    })
    expect(adjusted).toEqual([])
  })

  it('widens the chosen option and look card in forced colours and rings every pressed control in Highlight (M19)', () => {
    const narrow = Object.entries(WIDENED_IN_FORCED_COLOURS).flatMap(([file, selectors]) => {
      const rules = forcedColoursRules(readFrontend(file))
      return selectors
        .filter((selector) => !(Number.parseFloat(declarations(bodyOf(rules, selector)).get('border-width') ?? '') >= PRESSED_EDGE_MIN_PX))
        .map((selector) => `${file}: ${selector}`)
    })
    expect(narrow).toEqual([])
    const theme = forcedColoursRules(readFrontend(PLAY_THEME_CSS))
    const pressed = declarations(bodyOf(theme, PRESSED_ANYWHERE))
    expect(pressed.get('outline')).toBe('2px solid Highlight')
    expect(pressed.get('outline-offset')).toBe('2px')
    expect(declarations(bodyOf(theme, `${PRESSED_ANYWHERE}:focus-visible`)).get('outline-width')).toBe('4px')
    const swatchFocus = declarations(bodyOf(forcedColoursRules(readFrontend(EDITOR_CSS)), '.swatch:focus-visible'))
    expect(swatchFocus.get('outline-color')).toBe('CanvasText')
  })

  it("never hides the outline in Play's, the looks' or Auto's CSS: forced colours drop the box-shadow ring in its place (M19)", () => {
    const hidden = playCssFiles().flatMap((file) =>
      cssRules(fs.readFileSync(file, 'utf8'))
        .filter((rule) => hidesOutline(rule.body))
        .map((rule) => located(file, rule)),
    )
    expect(hidden).toEqual([])
  })

  it("lets a long look name wrap: the status column shrinks to 40% and the card's text breaks (M73)", () => {
    const rules = cssRules(readFrontend(LOOKS_CSS))
    const side = declarations(bodyOf(rules, '.headerSide'))
    expect(side.get('flex')).toBe('0 1 auto')
    expect(side.get('min-width')).toBe('0')
    expect(side.get('max-width')).toBe('40%')
    expect(declarations(bodyOf(rules, '.status')).get('overflow-wrap')).toBe('anywhere')
    const text = declarations(bodyOf(rules, '.cardText'))
    expect(text.get('min-width')).toBe('0')
    expect(text.get('overflow-wrap')).toBe('anywhere')
  })
})
