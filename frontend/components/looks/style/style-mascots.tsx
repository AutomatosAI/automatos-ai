import { AutoMascot } from '@/components/mascot/auto-mascot'
import { SHELLS } from '@/lib/looks/catalogue'
import { mascotSlots } from '@/lib/looks/style'
import type { Look } from '@/lib/looks/types'

import gallery from './gallery.module.css'
import styles from './style.module.css'

/**
 * PRD-260 §4 — the style sheet's mascot slots, ported from Automatos Play's
 * components/looks/style/StyleMascots.tsx: Auto's 12 states, each with the
 * ring he takes in this look (lib/mascot/ring.ts) and its colour in words.
 * Play drew stand-ins; here each slot is the real AutoMascot (T-003): his clip
 * for the browser, and every state's PNG under reduced motion. He is a
 * specimen, not the page's Auto, so his stages leave the page glow alone
 * (lightsPage={false}) and no caption speaks: the slot names the state.
 */

export const MASCOT_LABEL = 'Mascot slot · 12 states · ring drawn by the page'

export function StyleMascots({ look }: { readonly look: Look }) {
  return (
    <section className={styles.section} aria-label={MASCOT_LABEL}>
      <h2 className={styles.label}>{MASCOT_LABEL}</h2>
      <ul className={gallery.slots}>
        {mascotSlots(look).map((slot) => (
          <li key={slot.state} className={gallery.slotItem}>
            <div className={gallery.slotStage}>
              <AutoMascot state={slot.state} showCaption={false} lightsPage={false} />
            </div>
            <span className={gallery.slotName}>{slot.state}</span>
            <span className={gallery.slotWhen}>{slot.when}</span>
            <span className={gallery.slotColour}>
              ring {slot.tone} · {slot.ring}
            </span>
          </li>
        ))}
      </ul>
      <p className={styles.note}>
        Sizes: 220 desktop companion · 380 first open · 84 phone strip. Video: .webm (VP9 alpha) everywhere, .mp4
        (HEVC alpha) for Safari and iOS, the still (.png) under reduced motion. Auto&apos;s shell is a filter over the
        clip: {SHELLS[look.shell].name}.
      </p>
    </section>
  )
}
