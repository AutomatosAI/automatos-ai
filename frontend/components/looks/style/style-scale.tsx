import type { SwatchStyle } from '@/components/looks/swatch-style'
import { RADII, SPACING_STEPS, radiusToken } from '@/lib/looks/style'

import styles from './style.module.css'

/**
 * PRD-260 §4 — the style sheet's spacing scale and radii, ported from
 * Automatos Play's components/looks/style/StyleScale.tsx: each drawn at its
 * app/play-theme.css token's own value, in the accent. The value reaches the
 * CSS module as an inline --pl-* property typed by swatch-style.ts's
 * SwatchStyle, the one style-prop helper the looks pages share; only the
 * sheet draws steps and radii, so their two properties live here.
 */

export const SPACING_LABEL = 'Spacing · 4px base'
export const RADII_LABEL = 'Radii'

/** A spacing step drawn at a theme token's own value, e.g. var(--pl-space-4). */
export function stepStyle(token: string): SwatchStyle {
  return { '--pl-specimen-step': `var(${token})` }
}

/** A radius drawn at a theme token's own value, e.g. var(--pl-radius-card). */
export function radiusStyle(token: string): SwatchStyle {
  return { '--pl-specimen-radius': `var(${token})` }
}

export function StyleScale() {
  return (
    <>
      <section className={styles.section} aria-label={SPACING_LABEL}>
        <h2 className={styles.label}>{SPACING_LABEL}</h2>
        <ul className={styles.scale}>
          {SPACING_STEPS.map((step) => (
            <li key={step.token} className={styles.scaleItem}>
              <span className={styles.step} style={stepStyle(step.token)} aria-hidden="true" />
              {step.px}
            </li>
          ))}
        </ul>
      </section>
      <section className={styles.section} aria-label={RADII_LABEL}>
        <h2 className={styles.label}>{RADII_LABEL}</h2>
        <ul className={styles.scale}>
          {RADII.map((radius) => (
            <li key={radius.kind} className={styles.scaleItem}>
              <span className={styles.radius} style={radiusStyle(radiusToken(radius.kind))} aria-hidden="true" />
              {radius.label}
            </li>
          ))}
        </ul>
      </section>
    </>
  )
}
