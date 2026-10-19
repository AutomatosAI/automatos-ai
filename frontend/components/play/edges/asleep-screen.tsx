'use client'

import { useEffect, useRef } from 'react'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { useOnline } from '@/hooks/use-online'
import { useRecheckCountdown } from '@/hooks/use-recheck-countdown'
import type { MascotName } from '@/lib/mascot/types'
import { refocus } from '@/lib/play/focus'

import styles from './edges.module.css'
import { StartCommand } from './start-command'
import { EdgeWordmark } from './wordmark'

/**
 * PRD-260 §4 — "Automatos is asleep" (Ember mock 07), ported from Automatos
 * Play's components/edges/AsleepScreen.tsx: the whole window, the glow out,
 * Auto sleeping, the detail of what didn't answer, Check again, and the
 * automatic recheck counting down. Play's frame shows it in place of the page
 * while GET /health fails (asleep-gate.tsx). In the local edition it names the
 * command that starts Automatos; the hosted edition has none to offer. While a
 * check is in flight it says Checking… (M94); while the browser is offline it
 * says so, in place of either (M83, P260-FX-21). It focuses its heading as it
 * opens, since the swap took focus away with the page (P260-FX-22, M21).
 */

export const ASLEEP_TITLE = 'Automatos is asleep'
export const NOT_CONNECTED = 'Not connected'
export const CHECK_AGAIN = 'Check again'
export const CHECKING = 'Checking…'
export const HOSTED_LINE = "Automatos isn't answering right now. Nothing's broken on your side: this page comes back as soon as it answers."
export const OFFLINE_LINE = "You're offline. Check your internet connection: this page comes back as soon as Automatos answers."

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
  /** A check (the ask or one of its retries) is in flight. */
  readonly checking: boolean
  /** How often it asks on its own. */
  readonly recheckSeconds: number
  /** The local edition: show the command that starts Automatos. */
  readonly local: boolean
}

/** The way back: offline says so; else the local edition's start command, or the hosted line. */
function WayBack({ local }: { readonly local: boolean }) {
  const online = useOnline()
  if (!online) return <p className={styles.text}>{OFFLINE_LINE}</p>
  return local ? <StartCommand /> : <p className={styles.text}>{HOSTED_LINE}</p>
}

export function AsleepScreen({ mascot, onClipEnded, detail, onCheck, checking, recheckSeconds, local }: AsleepScreenProps) {
  const { remaining, checkNow } = useRecheckCountdown(recheckSeconds, onCheck, checking)
  const title = useRef<HTMLHeadingElement>(null)
  useEffect(() => {
    refocus(title.current)
  }, [])
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
        <h1 ref={title} className={styles.title} tabIndex={-1}>
          {ASLEEP_TITLE}
        </h1>
        <WayBack local={local} />
        <div className={styles.actions}>
          <button type="button" className={styles.checkAgain} onClick={checkNow}>
            {CHECK_AGAIN}
          </button>
          <span className={styles.hint}>{checking ? CHECKING : countdownLine(remaining)}</span>
        </div>
        <p className={styles.hint}>{detail}</p>
      </main>
    </div>
  )
}
