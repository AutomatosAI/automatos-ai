'use client'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { useRecheckCountdown } from '@/hooks/use-recheck-countdown'
import type { MascotName } from '@/lib/mascot/types'

import styles from './edges.module.css'
import { StartCommand } from './start-command'
import { EdgeWordmark } from './wordmark'

/**
 * PRD-260 §4 — "Automatos is asleep" (Ember mock 07), ported from Automatos
 * Play's components/edges/AsleepScreen.tsx: the whole window, the glow out,
 * Auto sleeping, the detail of what didn't answer, Check again, and the
 * automatic recheck counting down. Play's frame shows it in place of the page
 * while GET /health fails (asleep-gate.tsx). In the local edition it names the
 * command that starts Automatos; the hosted edition has none to offer.
 */

export const ASLEEP_TITLE = 'Automatos is asleep'
export const NOT_CONNECTED = 'Not connected'
export const CHECK_AGAIN = 'Check again'
export const CHECKING = 'Checking…'
export const HOSTED_LINE = "Automatos isn't answering right now. Nothing's broken on your side: this page comes back as soon as it answers."

/** The automatic recheck, in words: "Checking again in 7 seconds." */
export function countdownLine(seconds: number): string {
  if (seconds <= 0) return CHECKING
  return `Checking again in ${seconds} ${seconds === 1 ? 'second' : 'seconds'}.`
}

/** What didn't answer, naming the API's host when it is known (the connection pill's host). */
export function asleepDetail(host?: string): string {
  return host ? `Automatos didn't answer at ${host}.` : "Automatos didn't answer."
}

export interface AsleepScreenProps {
  /** Auto's state from the reducer (useBackendMascot): sleeping. */
  readonly mascot: MascotName
  readonly onClipEnded?: (state: MascotName) => void
  /** What didn't answer (asleepDetail). */
  readonly detail: string
  /** Ask GET /health again. */
  readonly onCheck: () => void
  /** How often it asks on its own. */
  readonly recheckSeconds: number
  /** The local edition: show the command that starts Automatos. */
  readonly local: boolean
}

export function AsleepScreen({ mascot, onClipEnded, detail, onCheck, recheckSeconds, local }: AsleepScreenProps) {
  const { remaining, checkNow } = useRecheckCountdown(recheckSeconds, onCheck)
  return (
    <div className={styles.screen}>
      <header className={styles.top}>
        <EdgeWordmark quiet />
        <span className={styles.status} role="status">
          {NOT_CONNECTED}
        </span>
      </header>
      <main className={styles.body}>
        <AutoMascot state={mascot} onClipEnded={onClipEnded} size="large" showCaption={false} />
        <h1 className={styles.title}>{ASLEEP_TITLE}</h1>
        {local ? <StartCommand /> : <p className={styles.text}>{HOSTED_LINE}</p>}
        <div className={styles.actions}>
          <button type="button" className={styles.checkAgain} onClick={checkNow}>
            {CHECK_AGAIN}
          </button>
          <span className={styles.hint}>{countdownLine(remaining)}</span>
        </div>
        <p className={styles.hint}>{detail}</p>
      </main>
    </div>
  )
}
