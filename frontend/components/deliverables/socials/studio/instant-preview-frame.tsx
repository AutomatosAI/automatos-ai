'use client'

/**
 * PRD-251D US-D201 (E8) — the instant preview's frame: the server's page at the template's own
 * size, scaled to the width of the preview, labelled as a preview.
 *
 * The page is a workspace's own template, its script included, so it runs only here: in a frame
 * with sandbox="allow-scripts" and nothing else (no same origin: it reads no cookie, storage or
 * page of the app's; no forms, popups or top navigation), under its own content security policy,
 * which lets it fetch nothing. It takes no pointer events, so nothing in it can be clicked.
 */
import { useEffect, useRef, useState, type RefObject } from 'react'

import { INSTANT_PREVIEW_LABEL, INSTANT_PREVIEW_TITLE } from './instant-preview-model'

/** Exactly what the frame allows: its own scripts, nothing more. */
export const INSTANT_PREVIEW_SANDBOX = 'allow-scripts'

/** The element's width in pixels, followed as it changes (0 until it is laid out). */
function useBoxWidth(box: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const element = box.current
    if (!element) return undefined
    const measure = () => setWidth(element.clientWidth)
    measure()
    if (typeof ResizeObserver === 'undefined') return undefined
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [box])
  return width
}

interface InstantPreviewFrameProps {
  /** The page, its fonts and photos in place. */
  page: string
  /** The template's size the page is drawn at. */
  width: number
  height: number
}

export function InstantPreviewFrame({ page, width, height }: InstantPreviewFrameProps) {
  const box = useRef<HTMLDivElement>(null)
  const shown = useBoxWidth(box)
  return (
    <div ref={box} className="absolute inset-0 overflow-hidden">
      <iframe
        data-testid="socials-instant-preview"
        title={INSTANT_PREVIEW_TITLE}
        sandbox={INSTANT_PREVIEW_SANDBOX}
        referrerPolicy="no-referrer"
        srcDoc={page}
        className="pointer-events-none absolute left-0 top-0 origin-top-left border-0"
        style={{ width, height, transform: `scale(${shown / width})` }}
      />
      <span className="absolute left-2 top-2 rounded bg-background/80 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[.12em] text-foreground">
        {INSTANT_PREVIEW_LABEL}
      </span>
    </div>
  )
}
