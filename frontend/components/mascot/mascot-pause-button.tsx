'use client'

import styles from './auto-mascot.module.css'

/**
 * PRD-261 A-13b (F10, WCAG 2.2.2) — the Pause Auto control under Auto. A toggle
 * button with one name and aria-pressed (as the style sheet's "Play Auto's
 * states"), so a screen reader hears "Pause Auto, pressed" while he is paused
 * rather than a name that flips under the user. AutoMascot shows it wherever he
 * can move; under the OS reduced-motion setting nothing moves, so there is none.
 */

export const PAUSE_AUTO = 'Pause Auto'

export interface MascotPauseButtonProps {
  readonly paused: boolean
  readonly onToggle: () => void
}

export function MascotPauseButton({ paused, onToggle }: MascotPauseButtonProps) {
  return (
    <button type="button" className={styles.pause} aria-pressed={paused} onClick={onToggle}>
      <span className={styles.pauseIcon} aria-hidden="true" />
      {PAUSE_AUTO}
    </button>
  )
}
