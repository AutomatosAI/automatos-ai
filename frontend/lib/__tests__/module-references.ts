/**
 * PRD-260 K1, M96 — the files a frontend source names, for the Play guards
 * (prd260-play-guards.test.ts). Read through the TypeScript compiler API and the
 * stylesheet's own syntax, never by raw text, so a comment or a string that mentions
 * a path never counts (14 Oct: a docstring tripped PRD-259's text-grep guard):
 *
 * - module specifiers: import/export declarations, import(), require(), vi.mock() and
 *   its kin, import types;
 * - the files next/font/local loads: the `src` of a call to the module's default
 *   export, one path or a list of faces' `path` (app/play-fonts.ts);
 * - a stylesheet's references: @import, url() and CSS Modules' composes … from.
 */
import ts from 'typescript'

import { withoutComments } from './css-cascade'

const MOCKING_CALLS = new Set(['mock', 'doMock', 'importActual', 'importMock'])
const LOCAL_FONT_MODULE = 'next/font/local'
/** `@import 'x'`, `url(x)` quoted or bare (`@import url(x)` too), `composes: a from 'x'`; the path is group 2. */
const CSS_REFERENCES = [/@import\s+(['"])(.*?)\1/g, /url\(\s*(['"]?)(.*?)\1\s*\)/g, /composes\s*:[^;{}]*?\bfrom\s+(['"])(.*?)\1/g]

export function parseSource(fileName: string, source: string): ts.SourceFile {
  const kind = fileName.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  return ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, false, kind)
}

export function stringArgument(node: ts.Node | undefined): string | null {
  return node !== undefined && ts.isStringLiteralLike(node) ? node.text : null
}

function isLoaderCall(callee: ts.Expression): boolean {
  if (callee.kind === ts.SyntaxKind.ImportKeyword) return true
  if (ts.isIdentifier(callee)) return callee.text === 'require'
  return ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) &&
    callee.expression.text === 'vi' && MOCKING_CALLS.has(callee.name.text)
}

export function specifierOf(node: ts.Node): string | null {
  if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) return stringArgument(node.moduleSpecifier)
  if (ts.isImportEqualsDeclaration(node) && ts.isExternalModuleReference(node.moduleReference)) {
    return stringArgument(node.moduleReference.expression)
  }
  if (ts.isCallExpression(node) && isLoaderCall(node.expression)) return stringArgument(node.arguments[0])
  if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) return stringArgument(node.argument.literal)
  return null
}

/** Every module specifier in the source: import/export declarations, import(), require(), vi.mock(), import types. */
export function moduleSpecifiers(fileName: string, source: string): string[] {
  const found: string[] = []
  const visit = (node: ts.Node): void => {
    const specifier = specifierOf(node)
    if (specifier !== null) found.push(specifier)
    ts.forEachChild(node, visit)
  }
  visit(parseSource(fileName, source))
  return found
}

// --- next/font/local ------------------------------------------------------------------

export function propertyNameOf(name: ts.Node): string {
  return ts.isIdentifier(name) || ts.isStringLiteralLike(name) ? name.text : ''
}

/** The value of an object literal's property: `{ src: … }` → the `…`. */
function optionValue(options: ts.Node | undefined, name: string): ts.Expression | undefined {
  if (options === undefined || !ts.isObjectLiteralExpression(options)) return undefined
  const property = options.properties.find((each) => ts.isPropertyAssignment(each) && propertyNameOf(each.name) === name)
  return property !== undefined && ts.isPropertyAssignment(property) ? property.initializer : undefined
}

/** A `src` option's files: `src: './a.woff2'`, or `src: [{ path: './a.woff2', weight: '400' }, …]`. */
function srcPaths(src: ts.Expression | undefined): string[] {
  const single = stringArgument(src)
  if (single !== null) return [single]
  if (src === undefined || !ts.isArrayLiteralExpression(src)) return []
  return src.elements.flatMap((face) => stringArgument(optionValue(face, 'path')) ?? [])
}

/** The names the file gives next/font/local's default export: `import localFont from 'next/font/local'`. */
function localFontLoaders(file: ts.SourceFile): Set<string> {
  return new Set(
    file.statements.flatMap((statement) =>
      ts.isImportDeclaration(statement) && stringArgument(statement.moduleSpecifier) === LOCAL_FONT_MODULE
        ? (statement.importClause?.name?.text ?? [])
        : [],
    ),
  )
}

/** Every file a next/font/local call in the source loads. */
export function localFontPaths(fileName: string, source: string): string[] {
  const file = parseSource(fileName, source)
  const loaders = localFontLoaders(file)
  const found: string[] = []
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && loaders.has(node.expression.text)) {
      found.push(...srcPaths(optionValue(node.arguments[0], 'src')))
    }
    ts.forEachChild(node, visit)
  }
  visit(file)
  return found
}

// --- stylesheets ----------------------------------------------------------------------

/** Every file a stylesheet names, in pattern order (@import, url(), composes), comments out. */
export function cssReferences(css: string): string[] {
  const bare = withoutComments(css)
  return CSS_REFERENCES.flatMap((pattern) => Array.from(bare.matchAll(pattern), (match) => match[2] ?? ''))
}
