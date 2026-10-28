/**
 * PRD-264 EUAI-11a — measuring the AI label's words and hit-testing them, for the label's check (visibility.ts):
 * the words' boxes as the browser lays them out (a Range), the points the check hit-tests, and the stacks of
 * elements under those points, read once per check for both the cover check and the ground check (ground.ts).
 *
 * Hit-testing skips anything that is `pointer-events: none`: the label itself (it never takes a click), and an
 * overlay a theme could make click-through to hide behind. So for the probe only, in the same task (nothing
 * paints meanwhile), a transient constructed stylesheet makes everything hittable, `*, *::before, *::after`
 * with `pointer-events: auto !important` (theme CSS can't be `!important`: the server's scope check refuses
 * it), and is taken out in a `finally`; being constructed, it adds no element a `:has()` rule could notice.
 * Only an inline `!important` beats it: the host's own protected copy, which must not count as a cover or a
 * ground.
 *
 * The points: SAMPLE_POINTS_PER_LINE evenly across each line of the words, ends included, and
 * JITTERED_POINTS_PER_LINE more at random, new each check, so nothing laid out between known points goes
 * unseen for long.
 */

/** Hit-test points evenly across each line of the words, ends included. */
export const SAMPLE_POINTS_PER_LINE = 9
/** And this many more per line, at random places, new each check. */
export const JITTERED_POINTS_PER_LINE = 2
/** The transient sheet that makes everything hittable for the moment of the probe. */
export const AI_LABEL_PROBE_ATTR = 'data-ai-label-probe'

const PROBE_RULE = '*, *::before, *::after { pointer-events: auto !important; }'
/** Keeps a hit-test point this far inside a line's ends, and its random height inside the line's middle half. */
const LINE_END_INSET_PX = 2
const LINE_MIDDLE = { from: 0.25, span: 0.5 }

/** A box in viewport px, as a browser measures it. */
export interface Box {
  readonly left: number
  readonly top: number
  readonly right: number
  readonly bottom: number
  readonly width: number
  readonly height: number
}

/** The words as the browser lays them out: their whole box and each line's. */
export interface Measured {
  readonly box: Box
  readonly lines: readonly Box[]
}

/** The words' own boxes; null where the browser can't measure text. */
export function measureWords(words: Element): Measured | null {
  const range = words.ownerDocument.createRange()
  if (typeof range.getBoundingClientRect !== 'function' || typeof range.getClientRects !== 'function') return null
  range.selectNodeContents(words)
  const lines = Array.from(range.getClientRects()).filter((line) => line.width > 0 && line.height > 0)
  return { box: range.getBoundingClientRect(), lines }
}

/**
 * SAMPLE_POINTS_PER_LINE points evenly across a line of the words at its middle height, ends included,
 * and JITTERED_POINTS_PER_LINE more at random across it and inside its middle half: bounded, and new
 * each check.
 */
export function samplePoints(line: Box, random: () => number = Math.random): ReadonlyArray<readonly [number, number]> {
  const inset = Math.min(LINE_END_INSET_PX, line.width / 4)
  const span = line.width - 2 * inset
  const middle = line.top + line.height / 2
  const steps = SAMPLE_POINTS_PER_LINE - 1
  const even = Array.from({ length: SAMPLE_POINTS_PER_LINE }, (_, step) => [line.left + inset + (span * step) / steps, middle] as const)
  const jittered = Array.from(
    { length: JITTERED_POINTS_PER_LINE },
    () => [line.left + inset + span * random(), line.top + line.height * (LINE_MIDDLE.from + LINE_MIDDLE.span * random())] as const,
  )
  return [...even, ...jittered]
}

/** Whether the document takes constructed stylesheets (adoptedStyleSheets), which put no element in the page. */
function adoptsSheets(doc: Document): boolean {
  const view = doc.defaultView
  return Array.isArray(doc.adoptedStyleSheets) && typeof view?.CSSStyleSheet === 'function' && 'replaceSync' in view.CSSStyleSheet.prototype
}

/**
 * Runs `probe` with everything in the document hittable, then takes the sheet out in the `finally`, all in
 * this task, so no frame paints with it. A browser without constructed sheets gets a `<style>` at the end of
 * <html>, outside <head> and <body>, which the label's and the copy's observers watch.
 */
function withEverythingHittable<T>(doc: Document, probe: () => T): T {
  if (adoptsSheets(doc)) {
    const sheet = new (doc.defaultView as Window & typeof globalThis).CSSStyleSheet()
    sheet.replaceSync(PROBE_RULE)
    doc.adoptedStyleSheets = [...doc.adoptedStyleSheets, sheet]
    try {
      return probe()
    } finally {
      doc.adoptedStyleSheets = doc.adoptedStyleSheets.filter((adopted) => adopted !== sheet)
    }
  }
  const element = doc.createElement('style')
  element.setAttribute(AI_LABEL_PROBE_ATTR, '')
  element.textContent = PROBE_RULE
  doc.documentElement.appendChild(element)
  try {
    return probe()
  } finally {
    element.remove()
  }
}

/** A point the check hit-tested, and every element under it, topmost first. */
export interface PointStack {
  readonly x: number
  readonly y: number
  readonly hits: readonly Element[]
}

/**
 * The elements under each sample point of the words, topmost first, everything hittable for the moment of the
 * call; null where the browser can't measure the words or hit-test.
 */
export function stacksUnderWords(words: Element, measured: Measured | null): PointStack[] | null {
  const root = words.getRootNode() as Document | ShadowRoot
  if (measured === null || typeof root.elementsFromPoint !== 'function') return null
  const points = measured.lines.flatMap((line) => samplePoints(line))
  const doc = root.ownerDocument ?? (root as Document)
  return withEverythingHittable(doc, () => points.map(([x, y]) => ({ x, y, hits: root.elementsFromPoint(x, y) })))
}
