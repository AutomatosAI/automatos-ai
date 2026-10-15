/**
 * PRD-260 M01 — a small static reader of this repository's stylesheets for the cascade
 * guards. vitest runs with css: false (and jsdom has no cascade), so which rule wins on
 * Play's elements is proven from the stylesheets' text: their rules, selector lists and
 * specificity. Tailwind 3 emits globals.css's @layer blocks as plain rules, so between two
 * rules that reach one element, specificity decides (then source order, which nobody
 * should lean on). Enough CSS for our own files: comments, @media/@supports/@layer/
 * @container blocks (read through), other at-rules (skipped), and the specificity of
 * :global()/:local() (CSS Modules erase the wrapper), :not()/:is()/:has() and :where().
 */

export type Specificity = readonly [number, number, number]

export interface CssRule {
  /** The rule's selector list, each one trimmed with its whitespace collapsed. */
  readonly selectors: readonly string[]
  readonly body: string
}

const GROUPING_AT_RULE = /^@(?:media|supports|layer|container)\b/
const IDENT = /^-?[_a-zA-Z][\w-]*/
const COMBINATOR = /[\s>+~]/
const NONE: Specificity = [0, 0, 0]
const ID: Specificity = [1, 0, 0]
const CLASS: Specificity = [0, 1, 0]
const TYPE: Specificity = [0, 0, 1]
const ERASED_WRAPPERS = new Set(['global', 'local'])
const MATCHES_ANY = new Set(['not', 'is', 'has', 'matches'])

export function withoutComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, ' ')
}

/** The index of the bracket that closes the one at `open` (`{`, `(` or `[`). */
function closing(text: string, open: number): number {
  const pair: Record<string, string> = { '{': '}', '(': ')', '[': ']' }
  const opener = text[open] ?? ''
  const closer = pair[opener]
  let depth = 0
  for (let i = open; i < text.length; i++) {
    if (text[i] === opener) depth++
    else if (text[i] === closer && --depth === 0) return i
  }
  return text.length
}

/** Split on `separator` outside brackets: `.a, :is(.b, .c)` is two selectors. */
export function splitTopLevel(text: string, separator = ','): string[] {
  const parts: string[] = []
  let depth = 0
  let start = 0
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '(' || ch === '[') depth++
    else if (ch === ')' || ch === ']') depth--
    else if (ch === separator && depth === 0) {
      parts.push(text.slice(start, i))
      start = i + 1
    }
  }
  return [...parts, text.slice(start)].map((part) => part.trim().replace(/\s+/g, ' ')).filter(Boolean)
}

function readBlock(css: string, from: number, to: number): CssRule[] {
  const rules: CssRule[] = []
  let at = from
  while (at < to) {
    const open = css.indexOf('{', at)
    if (open < 0 || open >= to) break
    const close = closing(css, open)
    const raw = css.slice(at, open)
    // Statements such as `@tailwind base;` end before the rule that follows them.
    const prelude = raw.slice(raw.lastIndexOf(';') + 1).trim()
    if (GROUPING_AT_RULE.test(prelude)) rules.push(...readBlock(css, open + 1, close))
    else if (!prelude.startsWith('@')) rules.push({ selectors: splitTopLevel(prelude), body: css.slice(open + 1, close) })
    at = close + 1
  }
  return rules
}

/** Every style rule in the stylesheet, those inside @media, @supports and @layer included. */
export function cssRules(css: string): CssRule[] {
  const bare = withoutComments(css)
  return readBlock(bare, 0, bare.length)
}

/** The property names a rule's body declares. */
export function declaredProperties(body: string): string[] {
  return body
    .split(';')
    .map((declaration) => declaration.split(':')[0]?.trim() ?? '')
    .filter(Boolean)
}

/** The compound selector the rule styles: `.a > .b code:hover` → `code:hover`. */
export function lastCompound(selector: string): string {
  let depth = 0
  let start = 0
  for (let i = 0; i < selector.length; i++) {
    const ch = selector[i] ?? ''
    if (ch === '(' || ch === '[') depth++
    else if (ch === ')' || ch === ']') depth--
    else if (depth === 0 && COMBINATOR.test(ch)) start = i + 1
  }
  return selector.slice(start).trim()
}

/** The compound's element name, '' when it names none (`.title`, `:first-child`). */
export function elementOf(compound: string): string {
  return IDENT.exec(compound)?.[0].toLowerCase() ?? ''
}

/** Whether the selector names the class (`.title`, not `.titleRow`). */
export function namesClass(selector: string, className: string): boolean {
  return new RegExp(`\\.${className}(?![\\w-])`).test(selector)
}

function add(a: Specificity, b: Specificity): Specificity {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/** Negative when a loses to b, 0 on a tie, positive when a wins. */
export function compareSpecificity(a: Specificity, b: Specificity): number {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2]
}

function strongest(selectors: readonly string[]): Specificity {
  return selectors.map(specificity).reduce((best, next) => (compareSpecificity(next, best) > 0 ? next : best), NONE)
}

function functionalPseudo(name: string, argument: string): Specificity {
  if (name === 'where') return NONE
  if (ERASED_WRAPPERS.has(name)) return specificity(argument)
  if (MATCHES_ANY.has(name)) return strongest(splitTopLevel(argument))
  return CLASS
}

function pseudo(text: string): [Specificity, number] {
  if (text[1] === ':') return [TYPE, 2 + (IDENT.exec(text.slice(2))?.[0].length ?? 0)]
  const name = IDENT.exec(text.slice(1))?.[0] ?? ''
  const open = 1 + name.length
  if (text[open] !== '(') return [CLASS, open]
  const close = closing(text, open)
  return [functionalPseudo(name.toLowerCase(), text.slice(open + 1, close)), close + 1]
}

/** The specificity of the selector's first token, and the token's length. */
function token(text: string): [Specificity, number] {
  const ch = text[0]
  if (ch === '#') return [ID, 1 + (IDENT.exec(text.slice(1))?.[0].length ?? 0)]
  if (ch === '.') return [CLASS, 1 + (IDENT.exec(text.slice(1))?.[0].length ?? 0)]
  if (ch === '[') return [CLASS, closing(text, 0) + 1]
  if (ch === ':') return pseudo(text)
  const element = IDENT.exec(text)?.[0].length ?? 0
  return element > 0 ? [TYPE, element] : [NONE, 1]
}

/** Selectors Level 4 specificity: (ids, classes/attributes/pseudo-classes, elements/pseudo-elements). */
export function specificity(selector: string): Specificity {
  let total = NONE
  let rest = selector.trim()
  while (rest.length > 0) {
    const [weight, length] = token(rest)
    total = add(total, weight)
    rest = rest.slice(Math.max(length, 1))
  }
  return total
}
