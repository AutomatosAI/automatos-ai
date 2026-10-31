/**
 * PRD-268 §6.11.1, K17 (US-W04) — a grep test: no status is shown by colour alone in components/activity and
 * components/missions. Every status colour class sits beside a glyph and a word.
 *
 * What it reads. Each .tsx file there (tests aside) is parsed with TypeScript, and every class in a JSX element's
 * className is read. A status colour is a status role's colour (§6.12.2): a text-, bg-, border- (any side), ring-,
 * fill-, stroke- or outline- class in destructive, success, warning, warn-ink, info or running. A class behind a
 * variant (hover:, focus:, dark:) paints a control's state or pairs a light class, so it is not read.
 *
 * What shows a status. A colour shows a status when the className chooses it by the row's status: the class sits in
 * a ternary's branch, or after && or ||, whose condition reads a status (a status or state field, display_state, or
 * a flag named for one: isFailed, isBlocked, cantRun, needsAction, waiting, …). The colour then changes with the
 * status, so it shows it. A fixed colour (an error line always in red) shows no status by colour, its words do, and
 * a colour chosen by a rating, a selection or a threshold shows no status.
 *
 * The rule. Every such colour sits beside a glyph and a word: the element's grandparent element (an icon beside a
 * column of text), else its parent, else the element itself when its function returns it, draws a glyph (one of
 * §6.11.1's ● ○ ◆ ✓ ✕ ▸, an icon from lucide-react or named *Icon, or StatusLabel) and a word (text with a letter, or
 * an {expression} that renders one). The box is generous on purpose: any icon or glyph in it, and any rendered
 * expression, counts. So the rule catches a status mark with nothing beside it (a bare dot, a tinted word with no
 * glyph), not a weak pairing. A class read from a map or a variable is judged where it is drawn, which a grep of the
 * className can't see.
 */
import { readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

const FRONTEND = path.resolve(__dirname, '..', '..', '..')
const SCOPES = ['components/activity', 'components/missions'] as const

const ROLES = '(?:destructive|success|warning|warn-ink|info|running)'
const STATUS_CLASS = new RegExp(`^(?:text|bg|border(?:-[lrtbxyse])?|ring|fill|stroke|outline)-(?:${ROLES}|\\[hsl\\(var\\(--${ROLES}\\)\\)\\])(?:/\\d+)?$`)
/** The status fields and flags a condition that chooses a status colour reads. */
const STATUS_NAMES: ReadonlySet<string> = new Set(['status', 'state', 'display_state', 'waiting', 'requires_attention', 'cantRun', 'needsAction', 'stopped'])
const STATUS_FLAG = /^(?:is|has)(?:Success|Succeeded|Failed|Failure|Blocked|Running|Working|Done|Completed|Complete|Verified|Error|Stuck|Paused|Stopped|Cancelled|Waiting|Pending|Offline|Online)$/
/** §6.11.1's glyphs; — (stopped) is left out, as it is punctuation in running text everywhere. */
const GLYPHS = /[●○◆✓✕▸]/
const LETTER = /\p{L}/u
const GLYPH_COMPONENTS: ReadonlySet<string> = new Set(['StatusLabel', 'StatePill', 'RowStatusLabel'])
const CHOOSING_OPERATORS: ReadonlySet<ts.SyntaxKind> = new Set([ts.SyntaxKind.AmpersandAmpersandToken, ts.SyntaxKind.BarBarToken])

interface Finding {
  readonly file: string
  readonly line: number
  readonly classes: readonly string[]
}

function tsxFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name)
    if (statSync(full).isDirectory()) return name === '__tests__' ? [] : tsxFiles(full)
    return name.endsWith('.tsx') && !name.endsWith('.test.tsx') ? [full] : []
  })
}

function lucideNames(sf: ts.SourceFile): Set<string> {
  const names = new Set<string>()
  sf.statements.filter(ts.isImportDeclaration).forEach((decl) => {
    const from = decl.moduleSpecifier
    const bindings = decl.importClause?.namedBindings
    if (!ts.isStringLiteral(from) || from.text !== 'lucide-react' || !bindings || !ts.isNamedImports(bindings)) return
    bindings.elements.forEach((element) => names.add(element.name.text))
  })
  return names
}

function literalText(node: ts.Node): string | null {
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text
  if (ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) return node.text
  return null
}

function statusClasses(text: string): string[] {
  return text.split(/\s+/).filter((token) => STATUS_CLASS.test(token))
}

/** The conditions between a literal and its className attribute that choose it (ternary branches, &&, ||). */
function choosingConditions(literal: ts.Node, attribute: ts.Node): ts.Expression[] {
  const found: ts.Expression[] = []
  let child: ts.Node = literal
  for (let node = literal.parent; node && node !== attribute; child = node, node = node.parent) {
    if (ts.isConditionalExpression(node) && child !== node.condition) found.push(node.condition)
    if (ts.isBinaryExpression(node) && child === node.right && CHOOSING_OPERATORS.has(node.operatorToken.kind)) found.push(node.left)
  }
  return found
}

function readsAStatus(condition: ts.Expression): boolean {
  let reads = false
  const visit = (node: ts.Node): void => {
    if (ts.isIdentifier(node) && (STATUS_NAMES.has(node.text) || STATUS_FLAG.test(node.text))) reads = true
    ts.forEachChild(node, visit)
  }
  visit(condition)
  return reads
}

function hasJsx(node: ts.Node): boolean {
  if (ts.isJsxElement(node) || ts.isJsxSelfClosingElement(node) || ts.isJsxFragment(node)) return true
  return ts.forEachChild(node, hasJsx) ?? false
}

/** The literals an expression renders, its JSX elements' attributes aside. */
function renderedStrings(node: ts.Node): string[] {
  if (ts.isJsxAttributes(node)) return []
  const own = literalText(node)
  const below: string[] = []
  ts.forEachChild(node, (child) => {
    below.push(...renderedStrings(child))
  })
  return own === null ? below : [own, ...below]
}

interface Says {
  glyph: boolean
  word: boolean
}

function childExpression(node: ts.JsxExpression, says: Says): void {
  const expression = node.expression
  if (!expression) return
  const strings = renderedStrings(expression)
  if (strings.some((text) => GLYPHS.test(text))) says.glyph = true
  if (strings.some((text) => LETTER.test(text))) says.word = true
  if (!hasJsx(expression) && literalText(expression) === null) says.word = true
}

/** Whether `box` draws a glyph and a word: its text, its rendered expressions and its icons, attributes aside. */
function saysIt(box: ts.Node, sf: ts.SourceFile, icons: ReadonlySet<string>): Says {
  const says: Says = { glyph: false, word: false }
  const visit = (node: ts.Node): void => {
    if (ts.isJsxAttributes(node)) return
    if (ts.isJsxText(node)) {
      if (GLYPHS.test(node.text)) says.glyph = true
      if (LETTER.test(node.text)) says.word = true
    }
    if (ts.isJsxExpression(node) && (ts.isJsxElement(node.parent) || ts.isJsxFragment(node.parent))) childExpression(node, says)
    if (ts.isJsxSelfClosingElement(node) || ts.isJsxOpeningElement(node)) {
      const tag = node.tagName.getText(sf)
      if (icons.has(tag) || /Icon$/.test(tag) || GLYPH_COMPONENTS.has(tag)) says.glyph = true
    }
    ts.forEachChild(node, visit)
  }
  visit(box)
  return says
}

/** The element's grandparent element, else its parent, else the element itself. */
function besideBox(element: ts.Node): ts.Node {
  const elements: ts.Node[] = []
  for (let node = element.parent; node && elements.length < 2; node = node.parent) {
    if (ts.isJsxElement(node) || ts.isJsxFragment(node)) elements.push(node)
  }
  return elements[1] ?? elements[0] ?? element
}

function elementOf(attribute: ts.JsxAttribute): ts.Node {
  const owner = attribute.parent.parent
  return ts.isJsxOpeningElement(owner) ? owner.parent : owner
}

/** The status colours `attribute` chooses by a status. */
function chosenStatusColours(attribute: ts.JsxAttribute): string[] {
  const chosen: string[] = []
  const visit = (node: ts.Node): void => {
    const text = literalText(node)
    const classes = text === null ? [] : statusClasses(text)
    if (classes.length > 0 && choosingConditions(node, attribute).some(readsAStatus)) chosen.push(...classes)
    ts.forEachChild(node, visit)
  }
  if (attribute.initializer) visit(attribute.initializer)
  return chosen
}

/** Every status colour in `source` that no glyph and word sit beside. */
function colourAloneIn(file: string, source: string): Finding[] {
  const sf = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  const icons = lucideNames(sf)
  const findings: Finding[] = []
  const visit = (node: ts.Node): void => {
    if (ts.isJsxAttribute(node) && node.name.getText(sf) === 'className') {
      const classes = chosenStatusColours(node)
      const says = classes.length > 0 ? saysIt(besideBox(elementOf(node)), sf, icons) : null
      if (says && !(says.glyph && says.word)) {
        findings.push({ file, line: sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1, classes })
      }
    }
    ts.forEachChild(node, visit)
  }
  visit(sf)
  return findings
}

describe('no status by colour alone in components/activity and components/missions', () => {
  const files = SCOPES.flatMap((scope) => tsxFiles(path.join(FRONTEND, scope)))

  it('reads both folders', () => {
    SCOPES.forEach((scope) => expect(files.some((file) => file.startsWith(path.join(FRONTEND, scope) + path.sep)), scope).toBe(true))
    expect(files.length).toBeGreaterThan(40)
  })

  it('every status colour a status chooses sits beside a glyph and a word', () => {
    const findings = files.flatMap((file) => colourAloneIn(path.relative(FRONTEND, file), readFileSync(file, 'utf8')))
    expect(findings).toEqual([])
  })
})

describe('the rule itself', () => {
  const find = (jsx: string, imports = "import { XCircle } from 'lucide-react'\n") =>
    colourAloneIn('fixture.tsx', `${imports}export function Row({ task, rated }: any) {\n  return (\n${jsx}\n  )\n}\n`)

  it('finds a dot whose colour alone says the status', () => {
    expect(find(`<div><span className={task.status === 'failed' ? 'w-2 h-2 bg-destructive' : 'w-2 h-2 bg-muted'} /><span>{task.name}</span></div>`)).toEqual([
      { file: 'fixture.tsx', line: 4, classes: ['bg-destructive'] },
    ])
  })

  it('finds a coloured word with no glyph beside it', () => {
    expect(find(`<div><span className={cn('text-xs', task.isFailed && 'text-destructive')}>{task.name}</span></div>`)).toHaveLength(1)
  })

  it('passes the colour once a glyph and a word sit beside it: a §6.11.1 glyph or an icon', () => {
    expect(find(`<div><span aria-hidden="true">✕</span><span className={task.isFailed ? 'text-destructive' : ''}>{task.name}</span></div>`)).toEqual([])
    expect(find(`<div><XCircle className="w-3 h-3" /><span className={task.isFailed ? 'text-destructive' : ''}>Failed</span></div>`)).toEqual([])
  })

  it("leaves a fixed colour, a hover colour, and a colour chosen by what isn't a status", () => {
    expect(find(`<div><p className="text-destructive">{task.error}</p></div>`)).toEqual([])
    expect(find(`<div><button className={task.isFailed ? 'hover:text-destructive' : ''} /></div>`)).toEqual([])
    expect(find(`<div><span className={rated === -1 ? 'text-destructive' : 'text-muted-foreground'} /></div>`)).toEqual([])
  })

  it("reads className only, and never counts an attribute's strings as words", () => {
    expect(find(`<div title={task.isFailed ? 'text-destructive' : ''}><span className="w-2 h-2" /></div>`)).toEqual([])
    expect(find(`<div><span className={task.isFailed ? 'bg-destructive' : ''} /><b className="x">{' '}</b></div>`)).toHaveLength(1)
  })
})
