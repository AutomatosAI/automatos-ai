'use client'

/**
 * PRD-268 §6.5.1, §6.12.4 (US-W01b) — the chat's composer, with the newest open suggestion pinned above it in the
 * Auto thread.
 *
 * chat.tsx imports this under the composer's own name, so the Chat component (over the length limit) is untouched
 * and every place it draws the composer gains the line. It draws the pinned suggestion (pinned-suggestion.tsx),
 * then MultimodalInput with exactly the props it was given. Outside the Auto thread, or with no suggestion open,
 * the pinned suggestion draws nothing and the composer is exactly as before.
 *
 * PRD-268 US-W03a: it also sends each reply chip tapped in its chat (reply-chip-event.ts) through the sendMessage it
 * was given, as the person's own message, and does nothing while a turn is streaming. The box itself is unchanged.
 */
import {
  MultimodalInput as ComposerInput,
  type MultimodalInputProps,
} from '@/components/chatbot/multimodal-input'
import { PinnedSuggestion } from './pinned-suggestion'
import { useReplyChipsSent } from './reply-chip-event'

export function MultimodalInput(props: MultimodalInputProps) {
  useReplyChipsSent(props.chatId, props.status, props.sendMessage)
  return (
    <>
      <PinnedSuggestion chatId={props.chatId} />
      <ComposerInput {...props} />
    </>
  )
}
