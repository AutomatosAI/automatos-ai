/**
 * PRD-260 P260-FX-05 — the class lists of the frontend's sources, for the dark: guards.
 *
 * Read through the TypeScript compiler API, so a comment or JSX text that mentions a
 * class never counts. Two kinds of list:
 *
 * - every string literal (or template part) on its own: `'bg-info/10 text-info'`, a
 *   STATUS_CONFIG entry, a cva variant;
 * - every `className` attribute and every cn()/clsx()/twMerge() call as one list, its
 *   literals joined: classes that reach one element together even when they sit in
 *   separate strings (`cn('hover:bg-muted/50', isRunning && 'dark:bg-…')`). cva() is
 *   not joined: its variants never apply together.
 */
import fs from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

export const FRONTEND = path.resolve(__dirname, '../..')
export const SCANNED_DIRS = ['app', 'components', 'lib', 'hooks', 'contexts']
const SKIP_DIRS = new Set(['__tests__', 'node_modules', '.next', 'out', 'coverage'])
const SOURCE_FILE = /\.(ts|tsx)$/
const CLASS_HELPERS = new Set(['cn', 'clsx', 'twMerge'])

export interface SourceFile {
  /** The path from frontend/, with forward slashes. */
  readonly rel: string
  readonly text: string
}

function walk(dir: string): string[] {
  if (!fs.existsSync(dir)) return []
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return SKIP_DIRS.has(entry.name) ? [] : walk(full)
    return SOURCE_FILE.test(entry.name) ? [full] : []
  })
}

/** Every .ts/.tsx file under the scanned folders, tests excluded. */
export function scannedSources(): SourceFile[] {
  return SCANNED_DIRS.flatMap((dir) => walk(path.join(FRONTEND, dir))).map((file) => ({
    rel: path.relative(FRONTEND, file).split(path.sep).join('/'),
    text: fs.readFileSync(file, 'utf8'),
  }))
}

function literalText(node: ts.Node): string | null {
  if (ts.isStringLiteralLike(node) || ts.isTemplateLiteralToken(node)) return node.text
  return null
}

/** The text of every string literal and template part at or under `node`. */
function literalsUnder(node: ts.Node): string[] {
  const texts: string[] = []
  const visit = (child: ts.Node): void => {
    const text = literalText(child)
    if (text !== null) texts.push(text)
    ts.forEachChild(child, visit)
  }
  visit(node)
  return texts
}

function isClassNameAttribute(node: ts.Node): boolean {
  return ts.isJsxAttribute(node) && ts.isIdentifier(node.name) && node.name.text === 'className'
}

function isClassHelperCall(node: ts.Node): boolean {
  return ts.isCallExpression(node) && ts.isIdentifier(node.expression) && CLASS_HELPERS.has(node.expression.text)
}

function classTokens(texts: readonly string[]): string[] {
  return texts.flatMap((text) => text.split(/\s+/)).filter(Boolean)
}

function parse(source: string, fileName: string): ts.SourceFile {
  const kind = fileName.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  return ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, false, kind)
}

/** The class lists the nodes that `pick` accepts carry: each node's literals as one list. */
function listsOf(source: string, fileName: string, pick: (node: ts.Node) => string[] | null): string[][] {
  const lists: string[][] = []
  const visit = (node: ts.Node): void => {
    const texts = pick(node)
    if (texts) lists.push(classTokens(texts))
    ts.forEachChild(node, visit)
  }
  visit(parse(source, fileName))
  return lists.filter((list) => list.length > 0)
}

/** Each string literal's classes, on its own. */
export function literalClassLists(source: string, fileName: string): string[][] {
  return listsOf(source, fileName, (node) => {
    const text = literalText(node)
    return text === null ? null : [text]
  })
}

/** Each literal's classes, then each className attribute's and class helper call's, joined. */
export function classLists(source: string, fileName: string): string[][] {
  const joined = listsOf(source, fileName, (node) =>
    isClassNameAttribute(node) || isClassHelperCall(node) ? literalsUnder(node) : null,
  )
  return [...literalClassLists(source, fileName), ...joined]
}
