'use client'

import { type RefObject, useEffect } from 'react'

import { aiLabelHiddenReason } from '@/lib/ai-label/visibility'

/**
 * PRD-264 EUAI-11a — checks that the AI label can be seen, and says so whenever the answer may
 * have changed: on mount; when the theme changes (the style and the tone are classes and
 * attributes on <html>, and a theme's stylesheet is an element in <head>); when the window is
 * resized; and every RECHECK_INTERVAL_MS, for a stylesheet that finishes loading later.
 *
 * `firstCheckDelayMs` puts the first check off for a label that animates in (Auto's widget panel
 * fades in from opacity 0): checked at once it would read as hidden. 0, the default, checks on mount.
 */

/** A slow safety net: a theme's sheet that loads after mount changes no element the observer watches. */
export const RECHECK_INTERVAL_MS = 2000

/** The <html> attributes that carry the style (its classes), the tone and inline theme variables. */
const THEME_ATTRIBUTES = ['class', 'style', 'data-theme']

function observeThemeChanges(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: THEME_ATTRIBUTES })
  observer.observe(document.head, { childList: true, subtree: true, characterData: true })
  return () => observer.disconnect()
}

/** Runs `check` now, or after `delayMs`; returns what cancels a check still to come. */
function firstCheck(check: () => void, delayMs: number): () => void {
  if (delayMs <= 0) {
    check()
    return () => undefined
  }
  const timer = window.setTimeout(check, delayMs)
  return () => window.clearTimeout(timer)
}

/** Calls `onHiddenChange` with true when the label at `ref` can't be seen, false when it can. */
export function useAiLabelCheck(
  ref: RefObject<HTMLElement>,
  onHiddenChange: (hidden: boolean) => void,
  firstCheckDelayMs = 0,
): void {
  useEffect(() => {
    const check = () => onHiddenChange(aiLabelHiddenReason(ref.current, window) !== null)
    const cancelFirst = firstCheck(check, firstCheckDelayMs)
    const stopObserving = observeThemeChanges(check)
    const timer = window.setInterval(check, RECHECK_INTERVAL_MS)
    window.addEventListener('resize', check)
    return () => {
      cancelFirst()
      stopObserving()
      window.clearInterval(timer)
      window.removeEventListener('resize', check)
    }
  }, [ref, onHiddenChange, firstCheckDelayMs])
}
