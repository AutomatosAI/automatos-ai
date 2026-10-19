import styles from './edges.module.css'

/**
 * PRD-260 §4 — "Automatos Play" for the screens shown without Play's sidebar
 * (the asleep screen and the 404), ported from Automatos Play's
 * components/edges/Wordmark.tsx. Quiet (no accent) while nothing is running.
 */

export interface EdgeWordmarkProps {
  /** The asleep screen: no accent, so nothing on it is lit. */
  readonly quiet?: boolean
}

export function EdgeWordmark({ quiet = false }: EdgeWordmarkProps) {
  return (
    <div className={styles.wordmark} data-quiet={quiet ? 'true' : 'false'}>
      Automatos <span className={styles.wordmarkPlay}>Play</span>
    </div>
  )
}
