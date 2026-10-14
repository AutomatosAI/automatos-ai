/**
 * Asking again: Studio's regenerate and Play's Try again send the owner's last message anew,
 * and that message and everything after it leave the list first, so the new turn takes their
 * place.
 *
 * Why it lives here and not in `useChat` (./hooks): `useChat`'s own reload was memoised on the
 * messages alone. A turn's last change to the messages lands while it is still loading (an `e:`
 * frame's error, the last words), and the turn then ends without another, so that reload kept a
 * `sendMessage` from a loading render, which returns at once: the owner's message left the list
 * and nothing was sent. `useChatWithReceipts` builds `reload` from this on each render's own
 * `sendMessage` instead. `useChat` is far over the function-length limit, so it is not edited.
 */
import type { ChatMessage } from '@/types'

/** The owner's last message, and the messages that stay before it when it is asked again. */
export interface LastTurn {
  readonly kept: ChatMessage[]
  readonly content: string
}

/** The last turn to ask again, or null when the owner has said nothing yet. */
export function lastTurn(messages: readonly ChatMessage[]): LastTurn | null {
  const index = messages.map((m) => m.role).lastIndexOf('user')
  if (index < 0) return null
  return { kept: messages.slice(0, index), content: messages[index].content || '' }
}
