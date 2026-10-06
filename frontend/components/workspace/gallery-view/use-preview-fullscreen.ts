/**
 * usePreviewFullscreen (Gerard, 7 Oct)
 * ====================================
 *
 * A Deliverable's PDF or web page opens in the side panel; this lets the owner switch
 * its preview box to full screen and back. It uses the browser's Fullscreen API on
 * the box (Esc leaves it, as everywhere). Where the browser refuses or has no API
 * (Safari on an iPhone, a page framed without the permission), the box covers the
 * window instead (`fallback`), and Esc or the exit button leaves it.
 */

'use client'

import { useCallback, useEffect, useState, type RefObject } from 'react'

type FullscreenElement = HTMLElement & { webkitRequestFullscreen?: () => Promise<void> | void }
type FullscreenDocument = Document & {
  webkitFullscreenElement?: Element | null
  webkitExitFullscreen?: () => Promise<void> | void
}

const CHANGE_EVENTS = ['fullscreenchange', 'webkitfullscreenchange'] as const

function fullscreenElement(): Element | null {
  const doc = document as FullscreenDocument
  return doc.fullscreenElement ?? doc.webkitFullscreenElement ?? null
}

async function requestNative(el: FullscreenElement): Promise<boolean> {
  try {
    if (el.requestFullscreen) {
      await el.requestFullscreen()
      return true
    }
    if (el.webkitRequestFullscreen) {
      await el.webkitRequestFullscreen()
      return true
    }
  } catch {
    // Refused (no permission in a frame, or no user gesture): cover the window instead.
  }
  return false
}

export function usePreviewFullscreen(ref: RefObject<HTMLElement>) {
  const [native, setNative] = useState(false)
  const [fallback, setFallback] = useState(false)

  useEffect(() => {
    const sync = () => setNative(ref.current !== null && fullscreenElement() === ref.current)
    CHANGE_EVENTS.forEach((name) => document.addEventListener(name, sync))
    return () => CHANGE_EVENTS.forEach((name) => document.removeEventListener(name, sync))
  }, [ref])

  useEffect(() => {
    if (!fallback) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setFallback(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [fallback])

  const enter = useCallback(async () => {
    const el = ref.current as FullscreenElement | null
    if (!el) return
    if (!(await requestNative(el))) setFallback(true)
  }, [ref])

  const exit = useCallback(async () => {
    setFallback(false)
    const doc = document as FullscreenDocument
    if (doc.fullscreenElement && doc.exitFullscreen) await doc.exitFullscreen()
    else if (doc.webkitFullscreenElement && doc.webkitExitFullscreen) await doc.webkitExitFullscreen()
  }, [])

  return { isFullscreen: native || fallback, fallback, enter, exit }
}
