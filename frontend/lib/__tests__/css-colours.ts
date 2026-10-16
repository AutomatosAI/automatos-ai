/**
 * PRD-260 K2, M96 — the literal colours in Play's stylesheets and components, for the
 * Play guard (prd260-play-guards.test.ts). A literal colour is one the look can't change:
 *
 * - a #hex, or a colour function whose first channel is a number: rgb()/rgba(),
 *   hsl()/hsla(), hwb(), lab(), lch(), oklab(), oklch(), and color() after its colour
 *   space. hsl(var(--x)) and oklch(from var(--x) l c h) read the look: no literal;
 * - a named colour (white, rebeccapurple, …) in a declaration's value, and a CSS system
 *   colour (Canvas, Highlight, …) outside @media (forced-colors: active). Inside it a
 *   system colour is the point (P260-FX-19, M19): forced colours repaint the page in
 *   the system's own palette, never a look's.
 *
 * color-mix() holds a literal when an argument does: color-mix(in srgb, white 10%,
 * transparent) by its named colour, while color-mix(in srgb, var(--pl-accent) 30%,
 * transparent) reads the look. transparent and currentColor carry no colour of their
 * own, so they are never literals.
 */
import ts from 'typescript'

import { withoutComments } from './css-cascade'
import { parseSource, propertyNameOf, stringArgument } from './module-references'

/** #rgb, #rgba, #rrggbb, #rrggbbaa; not an HTML entity such as &#39;. */
const HEX_COLOUR = /(?<!&)#(?:[0-9a-f]{8}|[0-9a-f]{6}|[0-9a-f]{3,4})\b/gi
/** A colour function whose first channel is a literal number (or `none`); hsl(var(--x)) is not a literal. */
const COLOUR_FUNCTION = /\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\(\s*(?:[\d.]+|none\b)/gi
/** color() with a literal channel after its colour space: color(display-p3 1 .4 .1). */
const COLOUR_SPACE_FUNCTION = /\bcolor\(\s*[a-z][\w-]*\s+(?:[\d.]+|none\b)/gi

/** The 148 CSS named colours (CSS Color 4), lower case. */
const NAMED_COLOURS: ReadonlySet<string> = new Set(
  `aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet brown
  burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan
  darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid
  darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey darkturquoise darkviolet
  deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro
  ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki
  lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow
  lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray
  lightslategrey lightsteelblue lightyellow lime limegreen linen magenta maroon mediumaquamarine
  mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue mediumspringgreen
  mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite navy oldlace
  olive olivedrab orange orangered orchid palegoldenrod palegreen paleturquoise palevioletred
  papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red rosybrown royalblue
  saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue slategray slategrey
  snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow
  yellowgreen`.split(/\s+/),
)
/** The CSS system colours (CSS Color 4), lower case: forced colours paint with these. */
const SYSTEM_COLOURS: ReadonlySet<string> = new Set([
  'accentcolor', 'accentcolortext', 'activetext', 'buttonborder', 'buttonface', 'buttontext', 'canvas',
  'canvastext', 'field', 'fieldtext', 'graytext', 'highlight', 'highlighttext', 'linktext', 'mark',
  'marktext', 'selecteditem', 'selecteditemtext', 'visitedtext',
])
/** Outside forced colours, a system colour is as fixed as a named one. */
const FIXED_COLOURS: ReadonlySet<string> = new Set([...NAMED_COLOURS, ...SYSTEM_COLOURS])

/** A word in a value: not part of a hyphenated name, a class, an id or a number's unit, and not a function's name (tan()). */
const VALUE_WORD = /(?<![\w.#-])[a-z]+(?![\w(-])/gi
/** Quoted text and url(): never a colour. */
const NOT_A_COLOUR = /"[^"]*"|'[^']*'|url\([^)]*\)/g
/** A declaration's value: after `{` or `;`, up to `;` or `}`. A selector ends at `{`, so it never reads as one. */
const DECLARATION = /(?:^|[;{])\s*[\w-]+\s*:([^;{}]*)(?=[;}]|$)/g
const FORCED_COLOURS_BLOCK = /@media\s*\(\s*forced-colors\s*:\s*active\s*\)\s*\{/g
/** A style-object property or JSX attribute that paints: colour, background, border, shadow, fill, stroke, a custom property. */
const PAINT_NAME = /colou?r|background|border|outline|shadow|fill|stroke|caret|decoration|^--/i

/** Every #hex and colour function with literal channels in the text, comments included. */
export function literalColours(text: string): string[] {
  return [HEX_COLOUR, COLOUR_FUNCTION, COLOUR_SPACE_FUNCTION].flatMap((pattern) => Array.from(text.matchAll(pattern), (match) => match[0]))
}

/** The words of a value that name one of the colours, as written. */
function coloursNamed(value: string, names: ReadonlySet<string>): string[] {
  return Array.from(value.replace(NOT_A_COLOUR, ' ').matchAll(VALUE_WORD), (match) => match[0]).filter((word) =>
    names.has(word.toLowerCase()),
  )
}

/** The value of every declaration in the CSS. */
function declarationValues(css: string): string[] {
  return Array.from(css.matchAll(DECLARATION), (match) => match[1] ?? '')
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

export interface ForcedColoursSplit {
  /** The stylesheet without its forced-colours blocks. */
  readonly outside: string
  /** The body of each @media (forced-colors: active) block. */
  readonly inside: readonly string[]
}

/** The stylesheet, comments out, split into its @media (forced-colors: active) blocks and the rest. */
export function splitForcedColours(css: string): ForcedColoursSplit {
  const bare = withoutComments(css)
  const blocks = Array.from(bare.matchAll(FORCED_COLOURS_BLOCK), (match) => {
    const start = match.index ?? 0
    const open = start + match[0].length - 1
    return { start, open, close: blockEnd(bare, open) }
  })
  return {
    outside: blocks.reduceRight((rest, { start, close }) => rest.slice(0, start) + rest.slice(close + 1), bare),
    inside: blocks.map(({ open, close }) => bare.slice(open + 1, close)),
  }
}

/** Every literal colour in a stylesheet: #hex and colour functions anywhere, named and system colours in its declarations. */
export function stylesheetColours(css: string): string[] {
  const { outside, inside } = splitForcedColours(css)
  return [
    ...literalColours(css),
    ...declarationValues(outside).flatMap((value) => coloursNamed(value, FIXED_COLOURS)),
    ...inside.flatMap((block) => declarationValues(block)).flatMap((value) => coloursNamed(value, NAMED_COLOURS)),
  ]
}

/** The string a paint property or attribute holds: `{ color: 'x' }`, `fill="x"`, `stroke={'x'}`. */
function paintedString(node: ts.Node): string | null {
  if (ts.isPropertyAssignment(node) && PAINT_NAME.test(propertyNameOf(node.name))) return stringArgument(node.initializer)
  if (!ts.isJsxAttribute(node) || !PAINT_NAME.test(propertyNameOf(node.name))) return null
  const value = node.initializer
  return value !== undefined && ts.isJsxExpression(value) ? stringArgument(value.expression) : stringArgument(value)
}

/** Every literal colour in a component: #hex and colour functions anywhere, named and system colours in what it paints with. */
export function componentColours(fileName: string, source: string): string[] {
  const found: string[] = []
  const visit = (node: ts.Node): void => {
    const painted = paintedString(node)
    if (painted !== null) found.push(...coloursNamed(painted, FIXED_COLOURS))
    ts.forEachChild(node, visit)
  }
  visit(parseSource(fileName, source))
  return [...literalColours(source), ...found]
}
