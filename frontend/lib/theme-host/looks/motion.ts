/**
 * PRD-261 A-19d2 (plan §2.2.4, kit K-19c) — the reduced-motion marks the generic look runtime puts
 * on <html> under an installed theme with looks, for any theme's CSS to read: `data-theme-motion`
 * ('reduce' when the user asks for reduced motion, 'full' otherwise), for a selector that stops
 * loops and transitions, and `--theme-motion` (0 or 1), for CSS that scales motion with calc().
 * Theme-neutral names: no theme's own marks are set here. Leaving the theme takes both off.
 */

export const THEME_MOTION_ATTRIBUTE = 'data-theme-motion'
export const THEME_MOTION_VAR = '--theme-motion'

/** The attribute's two values. */
export const MOTION_REDUCE = 'reduce'
export const MOTION_FULL = 'full'

/** The variable's two values: a factor for calc(). */
const MOTION_OFF = '0'
const MOTION_ON = '1'

export function applyThemeMotion(root: HTMLElement, reducedMotion: boolean): void {
  root.setAttribute(THEME_MOTION_ATTRIBUTE, reducedMotion ? MOTION_REDUCE : MOTION_FULL)
  root.style.setProperty(THEME_MOTION_VAR, reducedMotion ? MOTION_OFF : MOTION_ON)
}

export function clearThemeMotion(root: HTMLElement): void {
  root.removeAttribute(THEME_MOTION_ATTRIBUTE)
  root.style.removeProperty(THEME_MOTION_VAR)
}
