import { RADII, SPACING_STEPS, radiusToken } from '@/lib/looks/style'

import { radiusStyle, stepStyle } from './specimen-style'
import styles from './style.module.css'

/**
 * PRD-260 §4 — the style sheet's spacing scale and radii, ported from
 * Automatos Play's components/looks/style/StyleScale.tsx: each drawn at its
 * app/play-theme.css token's own value, in the accent.
 */

export const SPACING_LABEL = 'Spacing · 4px base'
export const RADII_LABEL = 'Radii'

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
