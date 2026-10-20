'use client'

import type { HostToolCall } from '@automatos/theme-sdk'

import { toolLineLabel } from './labels'
import styles from './turn.module.css'

/**
 * PRD-260 §4 — one mono line per tool call (mocks 01 and 02), ported from
 * Automatos Play's components/chat/ToolLines.tsx onto the reply's tool calls
 * (the SDK's view model, PRD-261 A-14a; useChat keeps them on the reply through
 * upsertToolCall in lib/chat/tool-calls.ts): a spinner and "Searching the
 * web: '…'" while it runs, "✓ Searched the web · 3 results · 2.1 s" when it
 * is done, a failure in words, and "Waiting for you" on an ask for the owner's
 * click.
 */

export const TOOL_LINES_LABEL = 'Tools Auto used'

function ToolLineItem({ call, streaming }: { readonly call: HostToolCall; readonly streaming: boolean }) {
  const label = toolLineLabel(call, streaming)
  return (
    <li className={styles.toolLine} data-tone={label.tone}>
      {label.mark === null ? (
        <span className={styles.spinner} aria-hidden="true" />
      ) : (
        <span className={styles.mark} aria-hidden="true">
          {label.mark}
        </span>
      )}
      <span>{label.text}</span>
    </li>
  )
}

export interface ToolLinesProps {
  readonly calls: readonly HostToolCall[]
  /** The reply is being written: a running call spins; on a finished reply it was stopped. */
  readonly streaming: boolean
}

export function ToolLines({ calls, streaming }: ToolLinesProps) {
  if (calls.length === 0) return null
  return (
    <ul className={styles.toolLines} aria-label={TOOL_LINES_LABEL}>
      {calls.map((call) => (
        <ToolLineItem key={call.toolCallId} call={call} streaming={streaming} />
      ))}
    </ul>
  )
}
