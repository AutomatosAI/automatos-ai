'use client'

import { useState } from 'react'

import { useLook } from '@/components/looks/look-provider'
import { MOTION_NOTES } from '@/lib/looks/style'

import gallery from './gallery.module.css'
import styles from './style.module.css'

/**
 * PRD-260 §4 — the style sheet's Motion section, ported from Automatos Play's
 * components/looks/style/StyleMotion.tsx: the Ember motion notes, and a toggle
 * that plays them as small loops (a message rising, the ring pulsing, a
 * Deliverable landing, the glow fading). Under reduced motion the toggle is
 * off and says why; app/play-theme.css also stops every loop.
 */

export const MOTION_LABEL = 'Motion'
export const PLAY_MOTION = 'Play the motion'
export const REDUCED_MOTION_NOTE = 'Your system asks for reduced motion, so nothing plays here.'

function MotionDemos({ playing }: { readonly playing: boolean }) {
  return (
    <div className={gallery.demos} data-playing={playing} aria-hidden="true">
      <span className={gallery.demo}>
        <span className={gallery.demoMessage} />
        rise
      </span>
      <span className={gallery.demo}>
        <span className={gallery.demoRing} />
        ring
      </span>
      <span className={gallery.demo}>
        <span className={gallery.demoCard} />
        land
      </span>
      <span className={gallery.demo}>
        <span className={gallery.demoGlow} />
        glow
      </span>
    </div>
  )
}

export function StyleMotion() {
  const { reducedMotion } = useLook()
  const [wantsPlay, setWantsPlay] = useState(false)
  const playing = wantsPlay && !reducedMotion

  return (
    <section className={styles.section} aria-label={MOTION_LABEL}>
      <h2 className={styles.label}>{MOTION_LABEL}</h2>
      <ul className={styles.notes}>
        {MOTION_NOTES.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
      <MotionDemos playing={playing} />
      <button
        type="button"
        className={styles.toggle}
        aria-pressed={playing}
        disabled={reducedMotion}
        onClick={() => setWantsPlay((current) => !current)}
      >
        {PLAY_MOTION}
      </button>
      {reducedMotion && <p className={styles.note}>{REDUCED_MOTION_NOTE}</p>}
    </section>
  )
}
