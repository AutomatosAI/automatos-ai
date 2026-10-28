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
import { type CaptionTamper, TamperLog } from './protected-caption-tamper'
import { RECHECK_INTERVAL_MS } from './use-ai-label-check'

export { AI_LABEL_CAPTION_ATTR, captionPlaceInside }
export type { CaptionInset, CaptionPlace, CaptionTamper }
export { TAMPER_SUMMARY_MS, aiLabelCopyTamperSummary, aiLabelCopyTamperedWarning } from './protected-caption-tamper'

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
 * What a theme's code does to the copy is undone:
 * - removed from <body>, moved, its attributes or inline guard changed, or anything inside its
 *   shadow root changed: a MutationObserver on <body>'s children, the host's attributes and the
 *   whole shadow tree rebuilds it from scratch, in the same microtask checkpoint, up to
 *   MAX_HEALS_PER_WINDOW times a second (more would let a theme's own observer ping-pong with it and
 *   hang the page);
 * - hidden with hidePopover(): its `toggle` event rebuilds and re-shows it;
 * - covered by a popover, dialog or fullscreen element shown later (the top layer stacks in the
 *   order things enter it): `beforetoggle`, `toggle` (both caught at the document, capture phase)
 *   and `fullscreenchange` raise it again (hidePopover, then showPopover, which puts it last);
 * - and a watchdog checks it is whole, rebuilds it if not, and raises it: every RECHECK_INTERVAL_MS
 *   while nothing happens, and, from the first tampering until QUIET_BEFORE_BACKING_OFF_MS pass with
 *   none, every animation frame, before paint (requestAnimationFrame: once a frame, never a spin).
 *
 * The bound, precisely: while under attack the copy is checked and rebuilt once per frame, just
 * before that frame paints. Code of the theme's that removes it again after that, in the same frame
 * (its own MutationObserver, in the microtask checkpoint after our frame callback), can keep it off
 * that frame: winning that race would mean spinning with the theme's code, which hangs the page. That
 * code, like code that patches attachShadow, the popover methods or MutationObserver before the copy
 * exists, is beyond this: sandboxing a theme's code is not what this does, and the review reads that
 * code (docs/theme-host/css-layer-model.md). Every entry point fails safe: anything that throws is
 * logged, counted as tampering, and retried the next frame. Each kind of tampering logs one warning,
 * and while it goes on a count is logged at most every TAMPER_SUMMARY_MS (protected-caption-tamper.ts).
 * The shadow root is closed, so `host.shadowRoot` is null to a theme's code; tests reach it by spying
 * on attachShadow, so no handle sits on the element.
 */

const LOG_PREFIX = '[theme-host]'
/** Immediate repairs allowed in HEAL_WINDOW_MS; past them, the next frame repairs. */
export const MAX_HEALS_PER_WINDOW = 5
export const HEAL_WINDOW_MS = 1000
/** After this long with no tampering, the watchdog goes back from every frame to every RECHECK_INTERVAL_MS. */
export const QUIET_BEFORE_BACKING_OFF_MS = 3000

/** A `toggle` or `beforetoggle` event (ToggleEvent), read without relying on the type existing. */
type ToggleLike = Event & { readonly newState?: string }

export interface ProtectedCaption {
  readonly remove: () => void
}

/**
 * Shows the words in the browser's top layer, last in it; where it has none, they stay a fixed element.
 * Returns what the browser threw, or null: the words stay on screen as a fixed element either way.
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
  private readonly tampering = new TamperLog()
  private heals: readonly number[] = []
  private timer = 0
  private frame = 0
  private lastTamperAt = 0
  private stopped = false
  private raiseWarned = false

  constructor(
    private readonly doc: Document,
    private readonly text: string,
    private readonly place: () => CaptionPlace,
  ) {}

  start(): void {
    this.safely(() => this.build())
    this.safely(() => this.listen())
    this.safely(() => {
      this.timer = this.view().setInterval(() => this.safely(() => this.checkAndRepair()), RECHECK_INTERVAL_MS)
    })
  }

  stop(): void {
    this.stopped = true
    this.view().clearInterval(this.timer)
    if (this.frame !== 0) this.view().cancelAnimationFrame(this.frame)
    this.listeners.abort()
    this.safely(() => this.teardown())
  }

  private view(): Window {
    return this.doc.defaultView ?? window
  }

  /** Runs one step; anything it throws is logged, counted, and the next frame tries again. */
  private safely(step: () => void): void {
    try {
      step()
    } catch (error) {
      this.tampering.note('failed', Date.now(), error)
      this.watchEveryFrame()
    }
  }

  private build(): void {
    const parts = buildCaptionParts(this.doc, this.text, this.place())
    const built = new AbortController()
    this.parts = parts
    this.builtListeners = built
    parts.caption.addEventListener('toggle', (event) => this.safely(() => this.onOwnToggle(event as ToggleLike, parts)), {
      signal: built.signal,
    })
    const observer = new MutationObserver((records) => this.safely(() => this.onMutations(records, parts)))
    this.observer = observer
    observer.observe(this.doc.body, { childList: true })
    observer.observe(parts.host, { attributes: true })
    observer.observe(parts.root, { childList: true, subtree: true, characterData: true, attributes: true })
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
    const onTopLayer = (before: boolean) => (event: Event) => this.safely(() => this.onTopLayerChange(event as ToggleLike, before))
    this.doc.addEventListener('beforetoggle', onTopLayer(true), options)
    this.doc.addEventListener('toggle', onTopLayer(false), options)
    this.doc.addEventListener('fullscreenchange', () => this.safely(() => this.raise()), options)
    this.view().addEventListener('resize', () => this.safely(() => this.replace()), { signal: this.listeners.signal })
  }

  /** Another element entered the top layer: once it is in, put the words above it. */
  private onTopLayerChange(event: ToggleLike, before: boolean): void {
    if (event.newState !== 'open' || event.target === this.parts?.host) return
    if (before) queueMicrotask(() => this.safely(() => this.raise()))
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

  /** The watchdog's step: rebuild what isn't whole, and raise what is. */
  private checkAndRepair(): void {
    const parts = this.parts
    if (parts === null || !partsIntact(parts, this.doc, this.text)) {
      this.repair(parts?.host.parentNode === this.doc.body ? 'changed' : 'removed')
      return
    }
    this.raise()
  }

  /** From the first tampering until QUIET_BEFORE_BACKING_OFF_MS pass with none: check and repair every frame. */
  private watchEveryFrame(): void {
    if (this.stopped) return
    this.lastTamperAt = Date.now()
    if (this.frame === 0) this.frame = this.view().requestAnimationFrame(() => this.onFrame())
  }

  private onFrame(): void {
    this.frame = 0
    this.safely(() => this.checkAndRepair())
    const underAttack = Date.now() - this.lastTamperAt < QUIET_BEFORE_BACKING_OFF_MS
    if (!this.stopped && underAttack && this.frame === 0) this.frame = this.view().requestAnimationFrame(() => this.onFrame())
  }

  /** A repair an observer or a `toggle` asked for; past the per-second allowance, the next frame repairs. */
  private heal(kind: CaptionTamper): void {
    const now = Date.now()
    const recent = this.heals.filter((at) => now - at < HEAL_WINDOW_MS)
    if (recent.length >= MAX_HEALS_PER_WINDOW) {
      this.heals = recent
      this.tampering.note(kind, now)
      this.watchEveryFrame()
      return
    }
    this.heals = [...recent, now]
    this.repair(kind)
  }

  private repair(kind: CaptionTamper): void {
    this.tampering.note(kind, Date.now())
    this.watchEveryFrame()
    this.teardown()
    this.build()
  }

  /** Raises the words; a browser that won't (no top layer for them) leaves them a fixed element, warned once. */
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
