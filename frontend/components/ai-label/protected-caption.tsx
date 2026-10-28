'use client'

import { useEffect } from 'react'

import {
  AI_LABEL_CAPTION_ATTR,
  type CaptionInset,
  type CaptionParts,
  type CaptionPlace,
  buildCaptionParts,
  captionPlaceInside,
  partsIntact,
  placeCaption,
} from './protected-caption-parts'
import { RECHECK_INTERVAL_MS } from './use-ai-label-check'

export { AI_LABEL_CAPTION_ATTR, captionPlaceInside }
export type { CaptionInset, CaptionPlace }

/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — the AI label drawn again in the host's own layer,
 * where no theme's CSS reaches it, and kept there.
 *
 * While a theme keeps the host's AI label from being seen (use-ai-label-check.ts), the label draws
 * its words again here, and the chat keeps working: the person still reads "Auto is an AI". The
 * elements are protected-caption-parts.ts's: a closed shadow root on a guarded element of the host's
 * at the end of <body>, the words a manual popover in the browser's top layer, above every z-index
 * and out of reach of what an ancestor does to its descendants (opacity, filter, transform,
 * clip-path, overflow). Where the browser has no popover, they stay a fixed element.
 *
 * Nothing a theme's code does to the copy lasts:
 * - removed from <body>, moved, its attributes or inline guard changed, or anything inside its
 *   shadow root changed: a MutationObserver on <body>'s children, the host's attributes and the
 *   whole shadow tree rebuilds it at once, from scratch;
 * - hidden with hidePopover(): its `toggle` event rebuilds and re-shows it;
 * - covered by a popover, dialog or fullscreen element shown later (the top layer stacks in the
 *   order things enter it): `beforetoggle`, `toggle` (both caught at the document, capture phase)
 *   and `fullscreenchange` raise it again (hidePopover, then showPopover, which puts it last);
 * - and every RECHECK_INTERVAL_MS a watchdog checks it is whole, rebuilds it if not, and raises it,
 *   for whatever no event reports (a dialog opened in a shadow root of the theme's, say).
 * A theme that undoes every repair gets at most MAX_HEALS_PER_WINDOW immediate ones a second; the
 * watchdog repairs the rest, so the page never spins. Each kind of tampering logs one theme-host
 * warning. The shadow root is closed, so `host.shadowRoot` is null to a theme's code; tests reach it
 * by spying on attachShadow, so no handle sits on the element. Code a theme runs before the copy
 * exists (wrapping attachShadow or the popover methods, say) is beyond this: sandboxing a theme's
 * code is not what this does, and the review reads that code (docs/theme-host/css-layer-model.md).
 */

const LOG_PREFIX = '[theme-host]'
/** Immediate repairs allowed in HEAL_WINDOW_MS; past them, the watchdog repairs. */
export const MAX_HEALS_PER_WINDOW = 5
export const HEAL_WINDOW_MS = 1000

export type CaptionTamper = 'removed' | 'changed' | 'hidden'

const TAMPER_VERB: Readonly<Record<CaptionTamper, string>> = { removed: 'removed', changed: 'changed', hidden: 'hid' }

/** The warning the first tampering of each kind leaves in the console. */
export function aiLabelCopyTamperedWarning(kind: CaptionTamper): string {
  return `something in the page ${TAMPER_VERB[kind]} the AI label's protected copy; the host put it back`
}

/** A `toggle` or `beforetoggle` event (ToggleEvent), read without relying on the type existing. */
type ToggleLike = Event & { readonly newState?: string }

export interface ProtectedCaption {
  readonly remove: () => void
}

/**
 * Shows the words in the browser's top layer, last in it; where it has none, they stay a fixed element.
 * Returns what the browser threw, or null.
 */
function raiseToTop(caption: HTMLElement): unknown {
  if (typeof caption.showPopover !== 'function') return null
  try {
    if (typeof caption.hidePopover === 'function') caption.hidePopover()
    caption.showPopover()
    return null
  } catch (error) {
    return error ?? new Error('showPopover failed')
  }
}

/** Keeps one copy of the words on screen, whole and topmost, from start() until stop(). */
class CaptionKeeper {
  private parts: CaptionParts | null = null
  private observer: MutationObserver | null = null
  private builtListeners: AbortController | null = null
  private readonly listeners = new AbortController()
  private heals: readonly number[] = []
  private warned: readonly CaptionTamper[] = []
  private raiseWarned = false
  private timer = 0

  constructor(
    private readonly doc: Document,
    private readonly text: string,
    private readonly place: () => CaptionPlace,
  ) {}

  start(): void {
    this.build()
    this.listen()
    this.timer = this.view().setInterval(() => this.tick(), RECHECK_INTERVAL_MS)
  }

  stop(): void {
    this.view().clearInterval(this.timer)
    this.listeners.abort()
    this.teardown()
  }

  private view(): Window {
    return this.doc.defaultView ?? window
  }

  private build(): void {
    const parts = buildCaptionParts(this.doc, this.text, this.place())
    const built = new AbortController()
    parts.caption.addEventListener('toggle', (event) => this.onOwnToggle(event as ToggleLike, parts), { signal: built.signal })
    const observer = new MutationObserver((records) => this.onMutations(records, parts))
    observer.observe(this.doc.body, { childList: true })
    observer.observe(parts.host, { attributes: true })
    observer.observe(parts.root, { childList: true, subtree: true, characterData: true, attributes: true })
    this.parts = parts
    this.observer = observer
    this.builtListeners = built
    this.raise()
  }

  private teardown(): void {
    this.observer?.disconnect()
    this.builtListeners?.abort()
    this.parts?.host.remove()
    this.parts = null
  }

  private listen(): void {
    const options = { capture: true, signal: this.listeners.signal }
    this.doc.addEventListener('beforetoggle', (event) => this.onTopLayerChange(event as ToggleLike, true), options)
    this.doc.addEventListener('toggle', (event) => this.onTopLayerChange(event as ToggleLike, false), options)
    this.doc.addEventListener('fullscreenchange', () => this.raise(), options)
    this.view().addEventListener('resize', () => this.replace(), { signal: this.listeners.signal })
  }

  /** Another element entered the top layer: once it is in, put the words above it. */
  private onTopLayerChange(event: ToggleLike, before: boolean): void {
    if (event.newState !== 'open' || event.target === this.parts?.host) return
    if (before) queueMicrotask(() => this.raise())
    else this.raise()
  }

  private onOwnToggle(event: ToggleLike, parts: CaptionParts): void {
    if (parts === this.parts && event.newState === 'closed') this.heal('hidden')
  }

  private onMutations(records: readonly MutationRecord[], parts: CaptionParts): void {
    if (parts !== this.parts) return
    if (parts.host.parentNode !== this.doc.body) this.heal('removed')
    else if (records.some((record) => record.target !== this.doc.body)) this.heal('changed')
  }

  private tick(): void {
    const parts = this.parts
    if (parts === null || !partsIntact(parts, this.doc, this.text)) {
      this.repair(parts?.host.parentNode === this.doc.body ? 'changed' : 'removed')
      return
    }
    this.raise()
  }

  /** A repair the observers or a `toggle` asked for, unless too many came this second (the watchdog takes those). */
  private heal(kind: CaptionTamper): void {
    const now = Date.now()
    const recent = this.heals.filter((at) => now - at < HEAL_WINDOW_MS)
    this.heals = recent.length < MAX_HEALS_PER_WINDOW ? [...recent, now] : recent
    if (recent.length < MAX_HEALS_PER_WINDOW) this.repair(kind)
  }

  private repair(kind: CaptionTamper): void {
    if (!this.warned.includes(kind)) {
      console.warn(LOG_PREFIX, aiLabelCopyTamperedWarning(kind))
      this.warned = [...this.warned, kind]
    }
    this.teardown()
    this.build()
  }

  private raise(): void {
    if (this.parts === null) return
    const error = raiseToTop(this.parts.caption)
    if (error === null || this.raiseWarned) return
    console.warn(LOG_PREFIX, 'the AI label could not be shown in the top layer, so it stays a fixed caption', error)
    this.raiseWarned = true
  }

  /** The viewport changed: move the words, and drop the records of that move (it is the host's own). */
  private replace(): void {
    if (this.parts === null) return
    placeCaption(this.parts.caption, this.place())
    this.observer?.takeRecords()
  }
}

/** Draws `text` in the host's own layer at `place()`, read now and on every resize, and keeps it there until `remove`. */
export function mountProtectedCaption(doc: Document, text: string, place: () => CaptionPlace): ProtectedCaption {
  const keeper = new CaptionKeeper(doc, text, place)
  keeper.start()
  return { remove: () => keeper.stop() }
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
