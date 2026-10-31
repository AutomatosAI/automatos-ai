/**
 * PRD-268 US-C21f — the proactive hello theme the browser lane serves (e2e/browser/fixtures/hello-proactive,
 * specs/proactive-defaults.spec.ts) is valid as files, and holds to the SDK's names: its entry imports only the
 * shared modules, asks for the SDK version the proactive defaults came in (1.3.0) and the host's satisfies it,
 * takes from the SDK only names a built area exports, places each of the six proactive defaults, and hands each
 * only the props the SDK declares for it (packages/theme-sdk/areas/cards.d.ts), every required one included (a
 * missing or misspelt prop would throw in the lane). It never imports sonner, an `@/` path or `next/*`, and its
 * stylesheet styles only its own elements.
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import { satisfiesSdkRange } from '@/lib/theme-host/engines'
import { parseLookPreset } from '@/lib/theme-host/manifest'
import sharedModules from '@/lib/theme-host/shared-modules.json'

interface SurfaceFile {
  readonly status: 'built' | 'stub'
  readonly names: readonly string[]
}

interface Placement {
  readonly component: string
  readonly props: readonly string[]
}

interface DeclaredProps {
  readonly all: readonly string[]
  readonly required: readonly string[]
}

const FIXTURE = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-proactive')
const SURFACE_DIR = path.resolve(__dirname, '../surface')
const CARDS_DTS = path.resolve(__dirname, '../../../packages/theme-sdk/areas/cards.d.ts')
const SDK_SPECIFIER = '@automatos/theme-sdk'
const ENTRY = readFileSync(path.join(FIXTURE, 'entry.js'), 'utf8')
const STYLESHEET = readFileSync(path.join(FIXTURE, 'theme.css'), 'utf8')
const MANIFEST = JSON.parse(readFileSync(path.join(FIXTURE, 'theme.json'), 'utf8')) as Record<string, unknown>
const HTML_CLASS = 'hello-proactive'
/** The SDK version that added the proactive defaults (K12); the one before it lacks them. */
const PROACTIVE_SDK = '1.3.0'
const SDK_BEFORE_PROACTIVE = '1.2.0'
const PROACTIVE_DEFAULTS = ['BriefCard', 'SuggestionCard', 'StatusLabel', 'ReadinessList', 'CalibrationCard', 'ReplyChips'] as const
const JSX_CALLS: ReadonlySet<string> = new Set(['jsx', 'jsxs'])
const IMPORT = /^import\s+(?:\{([^}]*)\}|[\w$]+)\s+from\s+'([^']+)'/gm
/** A rule's selector list: whatever precedes a `{` outside a comment. */
const SELECTOR = /(^|})\s*([^{}]+?)\s*\{/g
const COMMENT = /\/\*[\s\S]*?\*\//g

function syntaxErrors(source: string, fileName: string): string[] {
  const output = ts.transpileModule(source, {
    reportDiagnostics: true,
    fileName,
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, allowJs: true },
  })
  return (output.diagnostics ?? []).map((diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'))
}

function importsOf(source: string): { specifier: string; names: string[] }[] {
  return Array.from(source.matchAll(IMPORT), (match) => ({
    specifier: match[2],
    names: (match[1] ?? '').split(',').map((name) => name.trim()).filter((name) => name.length > 0),
  }))
}

function builtNames(): string[] {
  return readdirSync(SURFACE_DIR)
    .filter((file) => file.endsWith('.json'))
    .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)
    .filter((surface) => surface.status === 'built')
    .flatMap((surface) => surface.names)
}

function selectorsOf(css: string): string[] {
  return Array.from(css.replace(COMMENT, '').matchAll(SELECTOR), (match) => match[2]).flatMap((list) => list.split(','))
}

function parsed(text: string, fileName: string, kind: ts.ScriptKind): ts.SourceFile {
  return ts.createSourceFile(fileName, text, ts.ScriptTarget.ES2022, true, kind)
}

/** An object literal's own property names (assignments, shorthands and methods). */
function propNames(props: ts.ObjectLiteralExpression, source: ts.SourceFile): string[] {
  return props.properties
    .filter((property) => ts.isPropertyAssignment(property) || ts.isShorthandPropertyAssignment(property) || ts.isMethodDeclaration(property))
    .map((property) => (property.name ? property.name.getText(source).replace(/^['"]|['"]$/g, '') : ''))
}

/** The component and props of a `jsx(Component, { … })` call, or null for any other node. */
function placementOf(node: ts.Node, source: ts.SourceFile): Placement | null {
  if (!ts.isCallExpression(node) || !ts.isIdentifier(node.expression) || !JSX_CALLS.has(node.expression.text)) return null
  const [type, props] = node.arguments
  if (!type || !ts.isIdentifier(type) || !props || !ts.isObjectLiteralExpression(props)) return null
  return { component: type.text, props: propNames(props, source) }
}

/** Every `jsx(Component, { … })` placement under `node`, outermost first. */
function placementsIn(node: ts.Node, source: ts.SourceFile): Placement[] {
  const own = placementOf(node, source)
  const nested = node.getChildren(source).flatMap((child) => placementsIn(child, source))
  return own ? [own, ...nested] : nested
}

/** The members of `interface <name>` in the SDK's declarations, and the ones without `?`. */
function declaredProps(source: ts.SourceFile, name: string): DeclaredProps {
  const declared = source.statements.find(
    (statement): statement is ts.InterfaceDeclaration => ts.isInterfaceDeclaration(statement) && statement.name.text === name,
  )
  const members = (declared?.members ?? []).filter(ts.isPropertySignature)
  return {
    all: members.map((member) => member.name.getText(source)),
    required: members.filter((member) => member.questionToken === undefined).map((member) => member.name.getText(source)),
  }
}

const ENTRY_SOURCE = parsed(ENTRY, 'entry.js', ts.ScriptKind.JS)
const CARDS_SOURCE = parsed(readFileSync(CARDS_DTS, 'utf8'), 'cards.d.ts', ts.ScriptKind.TS)
const PLACEMENTS = placementsIn(ENTRY_SOURCE, ENTRY_SOURCE)

describe('PRD-268 US-C21f — the proactive hello theme', () => {
  it('has its entry, manifest, stylesheet, preset and README, and the host takes its preset', () => {
    expect(readdirSync(FIXTURE).sort()).toEqual(['README.md', 'entry.js', 'presets', 'theme.css', 'theme.json'])
    expect(MANIFEST).toMatchObject({ entry: 'entry.js', styleSheet: 'theme.css', preset: 'presets/default.look.json', htmlClass: HTML_CLASS, base: 'studio' })
    const preset: unknown = JSON.parse(readFileSync(path.join(FIXTURE, 'presets', 'default.look.json'), 'utf8'))
    expect(parseLookPreset(preset).ok).toBe(true)
  })

  it("asks for the SDK the proactive defaults came in, and the host's SDK satisfies it", () => {
    const range = (MANIFEST.engines as Record<string, unknown> | undefined)?.sdk
    expect(satisfiesSdkRange(range, PROACTIVE_SDK)).toBe(true)
    expect(satisfiesSdkRange(range, sharedModules.sdkVersion)).toBe(true)
    expect(satisfiesSdkRange(range, SDK_BEFORE_PROACTIVE)).toBe(false)
  })

  it('parses as an ES module', () => {
    expect(syntaxErrors(ENTRY, 'entry.js')).toEqual([])
  })

  it('imports only modules the host shares', () => {
    const shared = sharedModules.modules.map((entry) => entry.specifier)
    const imported = importsOf(ENTRY).map((entry) => entry.specifier)
    expect(imported.length).toBeGreaterThan(0)
    expect(imported.filter((specifier) => !shared.includes(specifier))).toEqual([])
    expect(ENTRY).not.toMatch(/['"](sonner|next\/[^'"]*|@\/[^'"]*)['"]/)
  })

  it('takes from the SDK only names a built area exports, and places each of the six proactive defaults', () => {
    const fromSdk = importsOf(ENTRY).filter((entry) => entry.specifier === SDK_SPECIFIER).flatMap((entry) => entry.names)
    const built = builtNames()
    expect(fromSdk).toEqual(expect.arrayContaining([...PROACTIVE_DEFAULTS]))
    expect(fromSdk.filter((name) => !built.includes(name))).toEqual([])
    const placed = new Set(PLACEMENTS.map((placement) => placement.component))
    expect(PROACTIVE_DEFAULTS.filter((name) => !placed.has(name))).toEqual([])
  })

  it.each(PROACTIVE_DEFAULTS)('hands %s only the props the SDK declares for it, every required one included', (component) => {
    const declared = declaredProps(CARDS_SOURCE, `${component}Props`)
    const placements = PLACEMENTS.filter((placement) => placement.component === component)
    expect(declared.all.length).toBeGreaterThan(0)
    expect(placements.length).toBeGreaterThan(0)
    for (const placement of placements) {
      expect(placement.props.filter((prop) => !declared.all.includes(prop))).toEqual([])
      expect(declared.required.filter((prop) => !placement.props.includes(prop))).toEqual([])
    }
  })

  it("styles only its own elements, under its own class, and never the host's cards", () => {
    const selectors = selectorsOf(STYLESHEET)
    expect(selectors.length).toBeGreaterThan(0)
    expect(selectors.filter((selector) => !selector.trim().startsWith(`html.${HTML_CLASS} `))).toEqual([])
    expect(STYLESHEET).not.toMatch(/data-protected-surface|!important/)
  })
})
