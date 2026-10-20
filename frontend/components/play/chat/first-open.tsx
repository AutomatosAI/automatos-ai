'use client'

import type { MascotName } from '../../../lib/mascot/types'
import { AutoMascot } from '../../mascot/auto-mascot'

import styles from './first-open.module.css'

/**
 * PRD-260 §4 — the empty chat (mock 06), ported from Automatos Play's
 * components/chat/FirstOpen.tsx: Auto big (playing hello on the first open in
 * a browser session), "Hi, I'm Auto." and three examples that fill the
 * message box.
 */

export const HELLO_TITLE = "Hi, I'm Auto."
export const HELLO_LINE = 'Your Automatos, the fun way'
export const EXAMPLES = [
  'Plan my revision week',
  'Write captions for our bake sale',
  'Budget a weekend in Lisbon',
] as const

/** The headline says these in words already; other states (a nap, a snag) keep their caption. */
const SAID_BY_HEADLINE: ReadonlySet<MascotName> = new Set<MascotName>(['hello', 'idle', 'listening'])

export interface FirstOpenHeroProps {
  readonly state: MascotName
  readonly onClipEnded: (state: MascotName) => void
  readonly caption?: string
  readonly detail?: string
}

export function FirstOpenHero({ state, onClipEnded, caption, detail }: FirstOpenHeroProps) {
  return (
    <section className={styles.hero} aria-label="Say hello">
      <AutoMascot
        state={state}
        onClipEnded={onClipEnded}
        caption={caption}
        detail={detail}
        size="large"
        showCaption={!SAID_BY_HEADLINE.has(state)}
      />
      <h2 className={styles.title}>{HELLO_TITLE}</h2>
      <p className={styles.line}>{HELLO_LINE}</p>
    </section>
  )
}

export function ExampleChips({ onPick }: { readonly onPick: (text: string) => void }) {
  return (
    <ul className={styles.chips} aria-label="Examples to try">
      {EXAMPLES.map((example) => (
        <li key={example}>
          <button type="button" className={styles.chip} onClick={() => onPick(example)}>
            {example}
          </button>
        </li>
      ))}
    </ul>
  )
}
