/**
 * PRD-207 — the current exchange of a live call, read off Retell's running transcript (its `update` event): what the
 * member said last, and Auto's answer to it as far as it has been spoken.
 *
 * Auto's words count only when they come after the member's last utterance: while the member asks something new,
 * Auto's previous answer is not the answer to it (it would show under the new question in the thread).
 */

export interface LiveExchange {
  readonly userText: string
  readonly agentText: string
}

const NO_EXCHANGE: LiveExchange = Object.freeze({ userText: '', agentText: '' })

interface Utterance {
  readonly agent: boolean
  readonly text: string
}

function utterancesOf(transcript: unknown): readonly Utterance[] {
  if (!Array.isArray(transcript)) return []
  return transcript.map((entry) => {
    const line = (entry ?? {}) as { role?: unknown; content?: unknown }
    return { agent: line.role === 'agent', text: typeof line.content === 'string' ? line.content : '' }
  })
}

/** The member's last utterance and Auto's words after it ('' for either that isn't there yet). */
export function currentExchange(transcript: unknown): LiveExchange {
  const lines = utterancesOf(transcript)
  const last = lines[lines.length - 1]
  if (!last) return NO_EXCHANGE
  if (!last.agent) return Object.freeze({ userText: last.text, agentText: '' })
  const asked = [...lines].reverse().find((line) => !line.agent)
  return Object.freeze({ userText: asked?.text ?? '', agentText: last.text })
}
