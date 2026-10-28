import { PROTECTED_SURFACE_GUARD } from '@/components/theme-host/protected-surface'

/**
 * PRD-264 EUAI-11a — the elements of the AI label's protected copy (protected-caption.tsx keeps them
 * on screen): a host element the host appends to <body>, a closed shadow root on it, and the words
 * inside, a manual popover for the browser's top layer.
 *
 * - No selector in a document stylesheet matches inside a shadow root, and the root is closed, so a
 *   theme's code can't reach it through `host.shadowRoot` either.
 * - The words carry `all: initial` and their own values inline, every one `!important`, so nothing
 *   they would inherit from a theme's <html> or <body> reaches them; they read no custom property
 *   (a theme sets those), so their colours are the system's Canvas and CanvasText. They wrap inside
 *   the viewport (MAX_WIDTH), so a long agent name never pushes " is an AI" off-screen.
 * - The host element is in the page's DOM, so a theme rule can match it: it carries the approval
 *   cards' inline guard (PROTECTED_SURFACE_GUARD, every value `!important`), fixed, empty and
 *   click-through. An important declaration attached to an element beats every stylesheet
 *   declaration, and the server's scope check refuses `!important` and any selector that names
 *   the label (orchestrator/services/themes/css_scope.py).
 */

export const AI_LABEL_CAPTION_ATTR = 'data-ai-label-caption'

/** Where the caption sits in the viewport: px from each edge named; an edge left out is auto. */
export interface CaptionPlace {
  readonly top?: number
  readonly right?: number
  readonly bottom?: number
  readonly left?: number
}

/** An offset from an element's top-left corner, in px. */
export interface CaptionInset {
  readonly top: number
  readonly left: number
}

export interface CaptionParts {
  readonly host: HTMLElement
  readonly root: ShadowRoot
  readonly caption: HTMLElement
}

const IMPORTANT = 'important'
const EDGES = ['top', 'right', 'bottom', 'left'] as const
/** Above anything a page stacks, for a browser with no top layer. */
const TOPMOST_Z_INDEX = '2147483647'
/** Keeps a caption anchored to an element inside the viewport, and leaves this much room at each side. */
const VIEWPORT_MARGIN_PX = 8
const CAPTION_ROOM_PX = { width: 160, height: 16 }
/** The widest the words run before they wrap: the viewport, less a margin at each side. */
const MAX_WIDTH = `calc(100vw - ${2 * VIEWPORT_MARGIN_PX}px)`

/**
 * The host element: the cards' guard, but fixed, empty and click-through, and closing what else
 * would make an ancestor of a fixed element its containing block (will-change, perspective,
 * backdrop-filter, a size container).
 */
const HOST_GUARD: Readonly<Record<string, string>> = Object.freeze({
  ...PROTECTED_SURFACE_GUARD,
  position: 'fixed',
  top: '0px',
  left: '0px',
  width: '0px',
  height: '0px',
  'z-index': TOPMOST_Z_INDEX,
  'pointer-events': 'none',
  'will-change': 'auto',
  perspective: 'none',
  'backdrop-filter': 'none',
  'container-type': 'normal',
})

/** The words: nothing inherited (`all: initial` first), a tiny calm caption in the system's colours. */
const CAPTION_STYLE: Readonly<Record<string, string>> = Object.freeze({
  all: 'initial',
  display: 'block',
  position: 'fixed',
  width: 'auto',
  height: 'auto',
  'max-width': MAX_WIDTH,
  margin: '0px',
  padding: '0px',
  visibility: 'visible',
  opacity: '1',
  'pointer-events': 'none',
  'z-index': TOPMOST_Z_INDEX,
  'white-space': 'normal',
  'overflow-wrap': 'anywhere',
  'color-scheme': 'light dark',
  color: 'CanvasText',
  'text-shadow': '0 0 2px Canvas, 0 0 2px Canvas, 0 0 4px Canvas',
  font: '500 11px/16px system-ui, sans-serif',
})

function styleAll(element: HTMLElement, declarations: Readonly<Record<string, string>>): void {
  for (const [property, value] of Object.entries(declarations)) element.style.setProperty(property, value, IMPORTANT)
}

/** Puts the words at `place`, every edge set (auto where `place` names none). */
export function placeCaption(caption: HTMLElement, place: CaptionPlace): void {
  for (const edge of EDGES) {
    const offset = place[edge]
    caption.style.setProperty(edge, offset === undefined ? 'auto' : `${offset}px`, IMPORTANT)
  }
}

function captionElement(doc: Document, text: string, place: CaptionPlace): HTMLElement {
  const caption = doc.createElement('p')
  caption.setAttribute('role', 'note')
  caption.setAttribute('aria-label', text)
  caption.setAttribute('popover', 'manual')
  caption.textContent = text
  styleAll(caption, CAPTION_STYLE)
  placeCaption(caption, place)
  return caption
}

/** A fresh host, closed shadow root and words, appended to <body>; nothing shown in the top layer yet. */
export function buildCaptionParts(doc: Document, text: string, place: CaptionPlace): CaptionParts {
  const host = doc.createElement('div')
  host.setAttribute(AI_LABEL_CAPTION_ATTR, '')
  styleAll(host, HOST_GUARD)
  const root = host.attachShadow({ mode: 'closed' })
  const caption = captionElement(doc, text, place)
  root.appendChild(caption)
  doc.body.appendChild(host)
  return { host, root, caption }
}

/** Whether the parts are still as built: the host a child of <body>, the root holding only the words, unchanged. */
export function partsIntact(parts: CaptionParts, doc: Document, text: string): boolean {
  const { host, root, caption } = parts
  return host.parentNode === doc.body && root.childNodes.length === 1 && root.firstChild === caption &&
    caption.textContent === text && caption.getAttribute('aria-label') === text
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max))
}

/** `inset` inside the top-left corner of `element`, kept whole inside the viewport. */
export function captionPlaceInside(element: Element | null, inset: CaptionInset, view: Window): CaptionPlace {
  const box = element?.getBoundingClientRect() ?? { top: 0, left: 0 }
  return {
    top: clamp(box.top + inset.top, VIEWPORT_MARGIN_PX, view.innerHeight - VIEWPORT_MARGIN_PX - CAPTION_ROOM_PX.height),
    left: clamp(box.left + inset.left, VIEWPORT_MARGIN_PX, view.innerWidth - VIEWPORT_MARGIN_PX - CAPTION_ROOM_PX.width),
  }
}
