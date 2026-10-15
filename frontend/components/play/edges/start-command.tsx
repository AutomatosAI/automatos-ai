'use client'

import { useCallback, useEffect, useState } from 'react'

import styles from './edges.module.css'

/**
 * PRD-260 §4 — the local edition's way back on the asleep screen (Ember mock
 * 07), ported from Automatos Play's AsleepScreen: the one command that starts
 * Automatos on this computer, with Copy. The hosted edition has no command to
 * offer, so the asleep screen shows HOSTED_LINE there instead (AGENTS.md: both
 * editions keep working).
 *
 * The command is SETUP.md's day-to-day start, `make up` (P260-FX-21, M84): it
 * starts Automatos in the background, where `docker compose up` held the
 * terminal. START_HINT names the start for an install from the published
 * images (`make up` would build from source there) and for a machine without make.
 */

export const START_COMMAND = 'make up'
export const START_HINT = 'Installed with the published images? Run make up-images. No make? docker compose up -d'
export const COPY_LABEL = 'Copy'
export const COPIED_LABEL = 'Copied'
export const COPY_FAILED = "Couldn't copy it. Select the command and copy it yourself."
/** How long the button says Copied. */
export const COPIED_MS = 2000

type CopyState = 'idle' | 'copied' | 'failed'

function useCopy(text: string): { readonly state: CopyState; readonly copy: () => Promise<void> } {
  const [state, setState] = useState<CopyState>('idle')
  useEffect(() => {
    if (state !== 'copied') return undefined
    const timer = window.setTimeout(() => setState('idle'), COPIED_MS)
    return () => window.clearTimeout(timer)
  }, [state])
  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text)
      setState('copied')
    } catch (error) {
      console.error("[asleep] couldn't copy the start command:", error)
      setState('failed')
    }
  }, [text])
  return { state, copy }
}

export function StartCommand() {
  const { state, copy } = useCopy(START_COMMAND)
  return (
    <>
      <p className={styles.text}>
        Automatos isn&apos;t running on your computer right now. Open a terminal in your{' '}
        <code className={styles.inline}>automatos-ai</code> folder and start it:
      </p>
      <div className={styles.command}>
        <span className={styles.prompt} aria-hidden="true">
          $
        </span>
        <code className={styles.commandText}>{START_COMMAND}</code>
        <button type="button" className={styles.copy} onClick={() => void copy()}>
          {state === 'copied' ? COPIED_LABEL : COPY_LABEL}
        </button>
      </div>
      <p className={styles.hint}>{START_HINT}</p>
      {state === 'failed' && (
        <p className={styles.hint} role="alert">
          {COPY_FAILED}
        </p>
      )}
    </>
  )
}
