'use client'

import { useEffect } from 'react'

import { PROTECTED_SURFACE_GUARD } from '@/components/theme-host/protected-surface'

/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — the AI label drawn again in the host's own layer,
 * where no theme's CSS reaches it.
 *
 * While a theme keeps the host's AI label from being seen (use-ai-label-check.ts), the label draws
 * its words again here, and the chat keeps working: the person still reads "Auto is an AI". Three
 * things keep a theme's stylesheet off this copy:
 *
 * - The words sit in a shadow root of an element the host appends to <body> itself, outside the
 *   theme's page, frame and slots. No selector in a document stylesheet matches an element inside
 *   a shadow root. The words carry `all: initial` and their own values inline, so nothing they
 *   would inherit from a theme's <html> or <body> reaches them either; they read no custom
 *   property (a theme sets those), so their colours are the system's Canvas and CanvasText.
 * - That element is in the page's DOM, so a theme rule can match it: it carries the approval
 *   cards' inline guard (PROTECTED_SURFACE_GUARD, every value `!important`), fixed and
 *   click-through. An important declaration attached to an element beats every stylesheet
 *   declaration, important or layered (docs/theme-host/css-layer-model.md); the server's scope
 *   check refuses `!important` in theme CSS anyway, and any selector that names the label
 *   (orchestrator/services/themes/css_scope.py).
 * - The words are a manual popover shown in the browser's top layer: above every z-index, and out
 *   of reach of what an ancestor does to its descendants (opacity, filter, transform, clip-path,
 *   overflow). Where the browser has no popover, they stay a fixed element.
 *
 * The shadow root is open, so tests and the browser lane can read it; a theme's code could reach
 * it, but a theme's code in the page could remove any element, and that stays a review rule, as a
 * top-layer element of the theme's own shown later does (docs/theme-host/css-layer-model.md).
 */

export const AI_LABEL_CAPTION_ATTR = 'data-ai-label-caption'

/** Where the caption sits in the viewport: px from each edge named; an edge left out is auto. */
export interface CaptionPlace {
  readonly top?: number
  readonly right?: number
  readonly bottom?: number
  readonly left?: number
}

const IMPORTANT = 'important'
const LOG_PREFIX = '[theme-host]'
const EDGES = ['top', 'right', 'bottom', 'left'] as const
/** Above anything a page stacks, for a browser with no top layer. */
const TOPMOST_Z_INDEX = '2147483647'
/** Keeps a caption anchored to an element whole inside the viewport. */
const VIEWPORT_MARGIN_PX = 8
const CAPTION_ROOM_PX = { width: 160, height: 16 }

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

/**
 * The words: nothing inherited (`all: initial` first), a tiny calm caption in the system's colours. No
 * document rule reaches inside the shadow root anyway; the values are important as well, so the copy
 * holds even where a rule did match it (jsdom matches a document's rules inside shadow roots).
 */
const CAPTION_STYLE: Readonly<Record<string, string>> = Object.freeze({
  all: 'initial',
  display: 'block',
  position: 'fixed',
  width: 'auto',
  height: 'auto',
  margin: '0px',
  padding: '0px',
  visibility: 'visible',
  opacity: '1',
  'pointer-events': 'none',
  'z-index': TOPMOST_Z_INDEX,
  'white-space': 'nowrap',
  'color-scheme': 'light dark',
  color: 'CanvasText',
  'text-shadow': '0 0 2px Canvas, 0 0 2px Canvas, 0 0 4px Canvas',
  font: '500 11px/16px system-ui, sans-serif',
})

export interface ProtectedCaption {
  readonly remove: () => void
}

function styleAll(element: HTMLElement, declarations: Readonly<Record<string, string>>, priority = ''): void {
  for (const [property, value] of Object.entries(declarations)) element.style.setProperty(property, value, priority)
}

function placeCaption(caption: HTMLElement, place: CaptionPlace): void {
  for (const edge of EDGES) {
    const offset = place[edge]
    caption.style.setProperty(edge, offset === undefined ? 'auto' : `${offset}px`, IMPORTANT)
  }
}

/** Shows the words in the browser's top layer; where it has none, they stay a fixed element. */
function raiseToTopLayer(caption: HTMLElement): void {
  if (typeof caption.showPopover !== 'function') return
  try {
    caption.showPopover()
  } catch (error) {
    console.warn(LOG_PREFIX, 'the AI label could not be shown in the top layer, so it stays a fixed caption', error)
  }
}

function captionElement(doc: Document, text: string): HTMLElement {
  const caption = doc.createElement('p')
  caption.setAttribute('role', 'note')
  caption.setAttribute('aria-label', text)
  caption.setAttribute('popover', 'manual')
  caption.textContent = text
  styleAll(caption, CAPTION_STYLE, IMPORTANT)
  return caption
}

/** Draws `text` in the host's own layer at `place()`, read now and on every resize, until `remove`. */
export function mountProtectedCaption(doc: Document, text: string, place: () => CaptionPlace): ProtectedCaption {
  const host = doc.createElement('div')
  host.setAttribute(AI_LABEL_CAPTION_ATTR, '')
  styleAll(host, HOST_GUARD, IMPORTANT)
  const caption = captionElement(doc, text)
  placeCaption(caption, place())
  host.attachShadow({ mode: 'open' }).appendChild(caption)
  doc.body.appendChild(host)
  raiseToTopLayer(caption)
  const view = doc.defaultView
  const replace = () => placeCaption(caption, place())
  view?.addEventListener('resize', replace)
  return {
    remove: () => {
      view?.removeEventListener('resize', replace)
      host.remove()
    },
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max))
}

/** An offset from an element's top-left corner, in px. */
export interface CaptionInset {
  readonly top: number
  readonly left: number
}

/** `inset` inside the top-left corner of `element`, kept whole inside the viewport. */
export function captionPlaceInside(element: Element | null, inset: CaptionInset, view: Window): CaptionPlace {
  const box = element?.getBoundingClientRect() ?? { top: 0, left: 0 }
  return {
    top: clamp(box.top + inset.top, VIEWPORT_MARGIN_PX, view.innerHeight - VIEWPORT_MARGIN_PX - CAPTION_ROOM_PX.height),
    left: clamp(box.left + inset.left, VIEWPORT_MARGIN_PX, view.innerWidth - VIEWPORT_MARGIN_PX - CAPTION_ROOM_PX.width),
  }
}

export interface ProtectedAiCaptionProps {
  readonly text: string
  /** Where it sits, read on mount and on every resize: a stable function (module-level or memoised). */
  readonly place: () => CaptionPlace
}

/** The AI label's words in the host's own layer, for as long as this is mounted. */
export function ProtectedAiCaption({ text, place }: ProtectedAiCaptionProps): null {
  useEffect(() => mountProtectedCaption(document, text, place).remove, [text, place])
  return null
}
