'use client'

import { type RefObject, useEffect } from 'react'

import type { AiLabelId } from '@/lib/ai-label/store'
import { type AiLabelHiddenReason, aiLabelHiddenReason } from '@/lib/ai-label/visibility'

/**
 * PRD-264 EUAI-11a — checks that the AI label can be seen, and says so whenever the answer may
 * have changed: on mount; when the theme changes (the style and the tone are classes and
 * attributes on <html>, and a theme's stylesheet is an element in <head>); when the window is
 * resized; and every RECHECK_INTERVAL_MS, for a stylesheet that finishes loading later.
 *
 * A label that can't be seen is drawn again in the host's own layer (protected-caption.tsx), and
 * the first check that finds it hidden, or hidden another way, logs a theme-host warning that
 * names the label and how it is hidden, so whoever tries or reviews the theme sees what it did.
 *
 * The check fails closed: if it throws (a theme's code may have patched getComputedStyle, Range,
 * elementsFromPoint or a canvas), the label counts as hidden ('unchecked') and the warning carries
 * the error. If the observer can't be set up, the interval and resize checks still run.
 *
 * `firstCheckDelayMs` puts the first check off for a label that animates in (Auto's widget panel
 * fades in from opacity 0): checked at once it would read as hidden. 0, the default, checks on mount.
 */

/** A slow safety net: a theme's sheet that loads after mount changes no element the observer watches. */
export const RECHECK_INTERVAL_MS = 2000

const LOG_PREFIX = '[theme-host]'

/** The <html> attributes that carry the style (its classes), the tone and inline theme variables. */
const THEME_ATTRIBUTES = ['class', 'style', 'data-theme']

/** The warning a theme that hides a label leaves in the console. */
export function aiLabelHiddenWarning(label: AiLabelId, reason: AiLabelHiddenReason): string {
  if (reason === 'unchecked') {
    return `the ${label} AI label could not be checked (unchecked: something in the page broke what the check calls); the host draws it in its own layer instead`
  }
  return `the active theme hides the ${label} AI label (${reason}); the host draws it in its own layer instead`
}

function observeThemeChanges(onChange: () => void): () => void {
  try {
    const observer = new MutationObserver(onChange)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: THEME_ATTRIBUTES })
    observer.observe(document.head, { childList: true, subtree: true, characterData: true })
    return () => observer.disconnect()
  } catch (error) {
    console.warn(LOG_PREFIX, "the AI label's check could not watch for theme changes; it still runs on its interval", error)
    return () => undefined
  }
}

interface Checked {
  readonly reason: AiLabelHiddenReason | null
  readonly error: unknown
}

/** Why the label can't be seen, or null; a check that throws reads as 'unchecked' (hidden), with its error. */
function checkLabel(element: HTMLElement | null): Checked {
  try {
    return { reason: aiLabelHiddenReason(element, window), error: null }
  } catch (error) {
    return { reason: 'unchecked', error }
  }
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
  label: AiLabelId,
  onHiddenChange: (hidden: boolean) => void,
  firstCheckDelayMs = 0,
): void {
  useEffect(() => {
    let lastReason: AiLabelHiddenReason | null = null
    const check = () => {
      const { reason, error } = checkLabel(ref.current)
      if (reason !== null && reason !== lastReason) {
        const detail = error === null ? [] : [error]
        console.warn(LOG_PREFIX, aiLabelHiddenWarning(label, reason), ...detail)
      }
      lastReason = reason
      onHiddenChange(reason !== null)
    }
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
  }, [ref, label, onHiddenChange, firstCheckDelayMs])
}
