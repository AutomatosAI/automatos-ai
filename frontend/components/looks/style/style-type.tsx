import { TEXT_SPECIMENS, displayLine, displaySpecimens } from '@/lib/looks/style'
import type { Look } from '@/lib/looks/types'

import styles from './style.module.css'
import typeStyles from './type.module.css'

/**
 * PRD-260 §4 — the style sheet's Type section, ported from Automatos Play's
 * components/looks/style/StyleType.tsx: the display specimens in the look's
 * own type style and sizes, then the reading type (Geist in every look) with
 * its sizes, each set in its app/play-theme.css token.
 */

export const TYPE_LABEL = 'Type'

/** Display specimens in order: hero, status, title. */
const DISPLAY_CLASSES = [typeStyles.hero, typeStyles.status, typeStyles.title]

/** Each reading specimen is set in its own token. */
const TEXT_CLASSES: Readonly<Record<string, string | undefined>> = {
  '--pl-text-reply': typeStyles.reply,
  '--pl-text-body': typeStyles.body,
  '--pl-text-button-sm': typeStyles.button,
  '--pl-text-label': typeStyles.caps,
  '--pl-text-mono': typeStyles.mono,
}

export function StyleType({ look }: { readonly look: Look }) {
  return (
    <section className={styles.section} aria-label={TYPE_LABEL}>
      <h2 className={styles.label}>{TYPE_LABEL}</h2>
      <dl className={typeStyles.specimens}>
        {displaySpecimens(look).map((specimen, index) => (
          <div key={specimen.label} className={typeStyles.specimen}>
            <dt className={typeStyles.specimenLabel}>{specimen.label}</dt>
            <dd className={DISPLAY_CLASSES[index]}>{specimen.sample}</dd>
          </div>
        ))}
        {TEXT_SPECIMENS.map((specimen) => (
          <div key={specimen.label} className={typeStyles.specimen}>
            <dt className={typeStyles.specimenLabel}>{specimen.label}</dt>
            <dd className={TEXT_CLASSES[specimen.token]}>{specimen.sample}</dd>
          </div>
        ))}
      </dl>
      <p className={styles.note}>
        Display: {displayLine(look)}. Reading: Geist 400–700 and Geist Mono in every look.
      </p>
    </section>
  )
}
