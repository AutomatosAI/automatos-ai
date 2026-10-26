'use client'

/**
 * Theme voice — Studio's quick ask: the host's voice button (components/voice/voice-button.tsx) in the chat composer's
 * action row, beside Send (composer-actions.tsx), on every plan and edition. Play shows Studio's chat below its
 * minimum width, so this is the phone's voice too. Hold to talk: the browser hears the member, the words go to Auto
 * as a spoken turn on the conversation on screen (a new chat takes the id the turn returns; the host does that), and
 * the reply is spoken. A click no pointer made (a key, a screen reader) toggles instead (voice-press.ts). Live stays
 * the mode bar's chip (chat-mode-bar.tsx): Live is the call, this is the quick ask.
 *
 * - No hotkey: in Studio, Space scrolls the page and is typed into the box.
 * - Never inside the AI label's fence (multimodal-input.tsx): the button's own caption, which no theme can hide, and
 *   the spoken notice say Auto is an AI, so voice is never locked on the page's label.
 * - Its slot is Send's size and shape (h-10 w-10, rounded-2xl), so the row never moves. While the session is the
 *   button's, its caption and controls open above it, over the box, in a card of the theme's tokens, and the press
 *   stays where the finger is: the slot places the button by its own hooks (`data-voice-button`, `data-voice-state`,
 *   `--voice-button-*`), as a theme would. The fill is a tint of the primary colour, so Send stays the main action.
 * - Outside a voice provider (app/chat/layout.tsx mounts one around every chat) the button is drawn disabled, saying
 *   voice isn't available on the page; a browser that can't recognise speech says so on a press, and the box still
 *   takes typing.
 */
import { VoiceButton } from '@/components/voice/voice-button'

const SLOT_CLASS = [
  'relative h-10 w-10 shrink-0',
  '[--voice-button-radius:1rem] [--voice-button-bg:hsl(var(--primary)_/_0.12)] [--voice-button-fg:hsl(var(--primary))]',
  // The root sits on the slot, its press at the bottom right; what it shows while the session is its own grows up
  // and to the left, never wider than the card's limit.
  '[&>[data-voice-button=root]]:absolute [&>[data-voice-button=root]]:-bottom-1.5 [&>[data-voice-button=root]]:-right-1.5',
  '[&>[data-voice-button=root]]:z-20 [&>[data-voice-button=root]]:w-max [&>[data-voice-button=root]]:max-w-[16rem]',
  '[&>[data-voice-button=root]]:flex-col-reverse [&>[data-voice-button=root]]:items-end',
  '[&>[data-voice-button=root]]:rounded-2xl [&>[data-voice-button=root]]:p-1.5',
  // While the session is the button's: a card in the theme's tokens behind the caption and the controls.
  '[&>[data-voice-button=root]:not([data-voice-state=idle])]:bg-popover',
  '[&>[data-voice-button=root]:not([data-voice-state=idle])]:text-popover-foreground',
  '[&>[data-voice-button=root]:not([data-voice-state=idle])]:shadow-md',
  '[&>[data-voice-button=root]:not([data-voice-state=idle])]:ring-1',
  '[&>[data-voice-button=root]:not([data-voice-state=idle])]:ring-border',
].join(' ')

/** The composer's voice button: the quick ask, held, no hotkey, at Send's size (md: 2.5rem, Send's h-10). */
export function ComposerVoice() {
  return (
    <div className={SLOT_CLASS} data-composer-voice="">
      <VoiceButton mode="quick" press="hold" hotkey={null} size="md" />
    </div>
  )
}
