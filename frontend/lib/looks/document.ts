import { applyCssVars, type CssVars } from '@/lib/look-kit/css'

import { LOOK_VAR_NAMES, STUDIO_TOKEN_NAMES, lookToCssVars, lookToStudioTokens } from './engine'
import type { Look } from './types'

/**
 * PRD-260 K3 — the look on <html>: inline custom properties, which outrank
 * .studio and .studio.dark, set only while the style is play. Leaving Play
 * removes every one, so Classic and Studio are exactly as before.
 */

/** On <html> under Play: 'reduce' or 'full'; app/play-theme.css stops loops and transitions under 'reduce'. */
export const MOTION_ATTRIBUTE = 'data-pl-motion'
/** On <html> under Play: 0 under reduced motion, 1 otherwise, for CSS that scales motion with calc(). */
export const MOTION_VAR = '--pl-motion'

/** Play's variables and Studio's tokens for a look: what the pre-paint script and the provider set. */
export function lookDocumentVars(look: Look): CssVars {
  return { ...lookToCssVars(look), ...lookToStudioTokens(look) }
}

/** Every inline property the look owns on <html>, the motion variable included. */
export const LOOK_DOCUMENT_NAMES: readonly string[] = [...LOOK_VAR_NAMES, ...STUDIO_TOKEN_NAMES, MOTION_VAR]

export function applyLook(root: HTMLElement, look: Look): void {
  applyCssVars(root, lookDocumentVars(look))
}

export function applyMotion(root: HTMLElement, reducedMotion: boolean): void {
  root.setAttribute(MOTION_ATTRIBUTE, reducedMotion ? 'reduce' : 'full')
  root.style.setProperty(MOTION_VAR, reducedMotion ? '0' : '1')
}

/** Takes every property and attribute the look set off <html>; other inline styles stay. */
export function clearLook(root: HTMLElement): void {
  LOOK_DOCUMENT_NAMES.forEach((name) => root.style.removeProperty(name))
  root.removeAttribute(MOTION_ATTRIBUTE)
}
