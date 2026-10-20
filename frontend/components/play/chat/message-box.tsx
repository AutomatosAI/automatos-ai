'use client'

import { type FormEvent, type KeyboardEvent, forwardRef, useState } from 'react'

import { AiLabelFence } from '@/components/ai-label/ai-label-fence'

import styles from './composer.module.css'

/**
 * PRD-260 §4 — the message box (mocks 01 and 02), ported from Automatos Play's
 * components/chat/MessageBox.tsx. Enter sends, Shift+Enter adds a line (and an
 * IME composition is never cut short), Send becomes Stop while Auto answers.
 * Focus and blur tell Auto you're typing. The ref is the textarea: Try again
 * puts focus back in it as its snag box goes (P260-FX-22, M20). PRD-264 EUAI-11a: the host's
 * AI label fence disables it while a theme hides the label.
 */

export const PLACEHOLDER = 'Message Auto'

export interface MessageBoxProps {
  /** Auto is answering: Stop instead of Send. */
  readonly streaming: boolean
  /** False when nothing went out (a turn was still ending): the words come back. */
  readonly onSend: (text: string) => boolean
  readonly onStop: () => void
  readonly onFocus?: () => void
  readonly onBlur?: () => void
  readonly initialDraft?: string
  /** Focus the box as it appears (the first open, and after an example fills it). */
  readonly autoFocus?: boolean
}

export const MessageBox = forwardRef<HTMLTextAreaElement, MessageBoxProps>(function MessageBox(
  { streaming, onSend, onStop, onFocus, onBlur, initialDraft = '', autoFocus = false },
  ref,
) {
  const [draft, setDraft] = useState(initialDraft)
  const ready = draft.trim().length > 0

  const submit = () => {
    if (streaming || !ready) return
    if (onSend(draft)) setDraft('')
  }

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    submit()
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return
    event.preventDefault()
    submit()
  }

  return (
    <AiLabelFence>
      <form className={styles.box} onSubmit={onSubmit}>
        <textarea
          ref={ref}
          className={styles.input}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
          onFocus={onFocus}
          onBlur={onBlur}
          placeholder={PLACEHOLDER}
          aria-label={PLACEHOLDER}
          autoFocus={autoFocus}
          rows={1}
        />
        {streaming ? (
          <button type="button" className={styles.stop} onClick={() => onStop()}>
            <span className={styles.stopIcon} aria-hidden="true" />
            Stop
          </button>
        ) : (
          <button type="submit" className={styles.send} disabled={!ready}>
            Send
          </button>
        )}
      </form>
    </AiLabelFence>
  )
})
