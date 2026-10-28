/**
 * PRD-264 EUAI-11a — the layout a browser hands the AI label's check, for jsdom, which lays nothing out.
 *
 * jsdom applies <style> rules and inline styles in getComputedStyle (display, visibility, opacity,
 * colour, background-color, clip-path, filter, font-size and the box properties as declared) but
 * measures nothing: no element box, no Range box, no hit-testing, no top layer. So each is
 * installed here, explicitly:
 *
 * - an element's box (getBoundingClientRect) from its computed left, top, width and height, a
 *   label-sized default where those are unset, so a theme's `width: 0px` or `top: -500px` reaches
 *   the check the way a browser's layout would hand it over;
 * - the words' Range box, and its one line: the box of the element the Range starts in, unless a
 *   test says what the browser would measure (`layout.textBox`, for a squeezed or shifted line jsdom
 *   can't lay out);
 * - hit-testing (elementsFromPoint, on a document and on a shadow root): `layout.cover` when a test
 *   draws something over the words, then the words' element, then the label, then `layout.under` when a
 *   test lays something of the page's under the label, then the label's ancestors up to <html>, as a
 *   browser's paint stack lists them (the check reads the ground from it), skipping, as a browser does,
 *   any whose computed `pointer-events` is `none` (so the check's transient probe sheet, which makes
 *   everything hittable, decides whether a click-through cover is seen);
 * - the copy's closed shadow roots, read off a spy on attachShadow (no handle sits on the element);
 * - a popover stand-in (installPopover): showPopover and hidePopover keep a per-element state and
 *   fire `toggle` a microtask later with the state then, as a browser coalesces a hide and a show.
 */
import { vi } from 'vitest'

import { AI_LABEL_CAPTION_ATTR } from '@/components/ai-label/protected-caption'

export interface LayoutBox {
  readonly left: number
  readonly top: number
  readonly width: number
  readonly height: number
}

export interface LabelLayout {
  /** What a browser would measure for the words, where jsdom can't lay them out; null: the element's box. */
  textBox: LayoutBox | null
  /** An element drawn over everything; null: nothing is. */
  cover: Element | null
  /** Where `cover` is drawn: only at the points this says (a striped overlay); null: at every point. */
  coverAt: ((x: number, y: number) => boolean) | null
  /** An element of the page's laid out under the label, not around it; null: nothing is. */
  under: Element | null
  /** Where `under` lies: only at the points this says (a patch under part of the words); null: at every point. */
  underAt: ((x: number, y: number) => boolean) | null
}

type Hits = (this: Document | ShadowRoot, x: number, y: number) => Element[]

const WORDS = '[data-ai-label-text]'

function px(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value)
  return value.trim().endsWith('px') && Number.isFinite(parsed) ? parsed : fallback
}

export function rect({ left, top, width, height }: LayoutBox): DOMRect {
  return { x: left, y: top, left, top, width, height, right: left + width, bottom: top + height, toJSON: () => ({}) } as DOMRect
}

function boxOf(element: Element, fallback: LayoutBox): DOMRect {
  const style = window.getComputedStyle(element)
  return rect({
    left: px(style.left, fallback.left),
    top: px(style.top, fallback.top),
    width: px(style.width, fallback.width),
    height: px(style.height, fallback.height),
  })
}

/** The element's ancestors, nearest first, up to <html>. */
function ancestorsOf(element: Element | null): Element[] {
  const chain: Element[] = []
  for (let node = element?.parentElement ?? null; node !== null; node = node.parentElement) chain.push(node)
  return chain
}

/**
 * What a browser's hit-test would find at a point of the words, topmost first: the cover, the words, the label,
 * what lies under the label, and the label's ancestors; each once.
 */
function stackIn(root: Document | ShadowRoot, cover: Element | null, under: Element | null): Element[] {
  const words = Array.from(root.querySelectorAll(WORDS))
  const stack: Array<Element | null> = [cover, ...words.flatMap((word) => [word, word.parentElement, under, ...ancestorsOf(word.parentElement)])]
  const hittable = stack.filter((element): element is Element => element !== null && window.getComputedStyle(element).pointerEvents !== 'none')
  return hittable.filter((element, index) => hittable.indexOf(element) === index)
}

function defineOn(target: object, name: string, value: unknown): () => void {
  Object.defineProperty(target, name, { value, configurable: true, writable: true })
  return () => {
    delete (target as Record<string, unknown>)[name]
  }
}

/** Installs the label's layout with `fallback` as the default box; returns what a test sets, and what undoes it. */
export function installLabelLayout(fallback: LayoutBox): { readonly layout: LabelLayout; readonly uninstall: () => void } {
  const layout: LabelLayout = { textBox: null, cover: null, coverAt: null, under: null, underAt: null }
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    return boxOf(this, fallback)
  })
  vi.spyOn(Element.prototype, 'attachShadow')
  const hits: Hits = function (this: Document | ShadowRoot, x: number, y: number) {
    const covered = layout.coverAt === null || layout.coverAt(x, y)
    const under = layout.underAt === null || layout.underAt(x, y)
    return stackIn(this, covered ? layout.cover : null, under ? layout.under : null)
  }
  const textRect = function (this: Range) {
    const start = this.startContainer
    const element = start instanceof Element ? start : start.parentElement
    return layout.textBox ? rect(layout.textBox) : boxOf(element as Element, fallback)
  }
  const lineRects = function (this: Range) {
    return [textRect.call(this)]
  }
  const undo = [
    defineOn(Range.prototype, 'getBoundingClientRect', textRect),
    defineOn(Range.prototype, 'getClientRects', lineRects),
    defineOn(Document.prototype, 'elementsFromPoint', hits),
    defineOn(ShadowRoot.prototype, 'elementsFromPoint', hits),
  ]
  return { layout, uninstall: () => undo.forEach((step) => step()) }
}

/** Every shadow root attachShadow made since the spy went in, oldest first. */
function shadowRoots(): ShadowRoot[] {
  return vi
    .mocked(Element.prototype.attachShadow)
    .mock.results.filter((result) => result.type === 'return')
    .map((result) => result.value as ShadowRoot)
}

/** The host element of the copy in the host's own layer, or null while there is none. */
export function captionHost(): HTMLElement | null {
  const hosts = document.querySelectorAll<HTMLElement>(`[${AI_LABEL_CAPTION_ATTR}]`)
  if (hosts.length > 1) throw new Error(`${hosts.length} copies of the AI label in the host's layer`)
  return hosts[0] ?? null
}

/** The words of the copy on screen, inside its host's closed shadow root; throws while there are none. */
export function theCaption(): HTMLElement {
  const host = captionHost()
  const root = shadowRoots().find((candidate) => candidate.host === host)
  const caption = root?.querySelector<HTMLElement>('[role="note"]')
  if (!caption) throw new Error("the host drew no copy of the label in its own layer")
  return caption
}

export interface PopoverStandIn {
  /** Every element showPopover put in the (pretend) top layer, in order. */
  readonly raised: Element[]
  readonly uninstall: () => void
}

/** showPopover and hidePopover, for a DOM without a top layer: state per element, `toggle` coalesced a microtask later. */
export function installPopover(): PopoverStandIn {
  const raised: Element[] = []
  const open = new WeakMap<Element, boolean>()
  const pending = new WeakSet<Element>()
  const toggleLater = (element: Element, was: boolean) => {
    if (pending.has(element)) return
    pending.add(element)
    queueMicrotask(() => {
      pending.delete(element)
      const now = open.get(element) === true
      const event = Object.assign(new Event('toggle'), { oldState: was ? 'open' : 'closed', newState: now ? 'open' : 'closed' })
      element.dispatchEvent(event)
    })
  }
  const show = function (this: Element) {
    toggleLater(this, open.get(this) === true)
    open.set(this, true)
    raised.push(this)
  }
  const hide = function (this: Element) {
    if (open.get(this) !== true) return
    toggleLater(this, true)
    open.set(this, false)
  }
  const undo = [defineOn(HTMLElement.prototype, 'showPopover', show), defineOn(HTMLElement.prototype, 'hidePopover', hide)]
  return { raised, uninstall: () => undo.forEach((step) => step()) }
}

/** Lets the microtasks queued so far (mutation observers, toggles) run. */
export async function flushMicrotasks(): Promise<void> {
  for (let round = 0; round < 4; round += 1) await Promise.resolve()
}
