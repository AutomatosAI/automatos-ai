'use client'

/**
 * PRD-268 §6.12.6, K11, K17 (US-C21d) — BriefVoice: Listen, the host's control that reads the day's brief aloud.
 *
 * Core draws all of it, so no theme ever speaks or plays audio itself: Listen reads the brief's `spoken` text (the
 * greeting and the numbered row titles) in Auto's server voice, with the browser's speech for the rest once the server
 * refuses (use-brief-voice.ts), and Stop stops it. Nothing plays until Listen is pressed, whatever a page or a theme
 * does: no prop makes it play on mount. While the tab's voice session runs (a hold or a call, voice-provider.tsx) Listen is
 * held, and a session starting stops a reading. Listen, and while Auto reads the "AI voice assistant" caption and
 * Stop, sit in the protected surface `brief-voice` (the inline !important guard, and the server's CSS scope check
 * refusing any theme selector that names a protected surface), so no theme hides how to stop it by name.
 *
 * A theme places it and syncs to it: the root (`data-brief-voice="root"`, outside the guard) carries
 * `data-voice-speaking="true"` while Auto reads, which a theme's `▸ SPEAKING` line follows, and (SDK 1.4.0)
 * `data-voice-row="<index>"` while Auto reads one of the brief's rows, its index in the sections' order
 * (brief-voice-rows.ts); `onSpeakingRow` hears the same, the index or null, each time it changes. Screen readers hear
 * that Auto is reading. A failure inside it renders nothing (VoiceErrorBoundary); the card beneath keeps working.
 */
import { Volume2 } from 'lucide-react'

import { ProtectedSurface } from '@/components/theme-host/protected-surface'
import { Button } from '@/components/ui/button'
import { AiVoiceLabel } from '@/components/voice/AiVoiceLabel'
import { VoiceErrorBoundary } from '@/components/voice/VoiceErrorBoundary'
import { useOptionalHostVoice } from '@/components/voice/voice-provider'
import type { VoicePhase } from '@/lib/voice/engine'

import type { SpokenRow } from './brief-voice-rows'
import { useBriefVoice } from './use-brief-voice'
import { CONTROL_ICON_CLASS, LABELLED_CONTROL_CLASS, StopButton } from './voice-controls'

export const LISTEN = 'Listen'
export const LISTEN_TO_THE_BRIEF = 'Listen to the brief'
export const READING_THE_BRIEF = 'Auto is reading the brief'

/** The voice session's phases while it holds the microphone or Auto's voice. */
const SESSION_RUNNING: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['connecting', 'listening', 'thinking', 'speaking'])

export interface BriefVoiceProps {
  /** The brief's `spoken` text (Brief@1): what Listen reads. With none, Listen is disabled. */
  readonly spoken: string
  /**
   * Hears the brief row Auto starts reading (its index), and null once it reads none, each time that changes, from the
   * speech's own events; a handler that throws is logged and the reading goes on.
   */
  readonly onSpeakingRow?: (row: SpokenRow) => void
}

function ListenButton({ disabled, onListen }: { readonly disabled: boolean; readonly onListen: () => void }) {
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className={LABELLED_CONTROL_CLASS}
      disabled={disabled}
      onClick={onListen}
      aria-label={LISTEN_TO_THE_BRIEF}
    >
      <Volume2 aria-hidden className={CONTROL_ICON_CLASS} />
      {LISTEN}
    </Button>
  )
}

/** Listen, or, while Auto reads the brief, the caption and Stop. */
export function BriefVoice({ spoken, onSpeakingRow }: BriefVoiceProps) {
  const text = typeof spoken === 'string' ? spoken : ''
  const voice = useOptionalHostVoice()
  const sessionRunning = voice !== null && SESSION_RUNNING.has(voice.snapshot.phase)
  const { speaking, row, listen, stop } = useBriefVoice(text, sessionRunning, onSpeakingRow)
  return (
    <VoiceErrorBoundary>
      <div
        className="w-fit"
        data-brief-voice="root"
        data-voice-speaking={speaking ? 'true' : undefined}
        data-voice-row={row === null ? undefined : String(row)}
      >
        <ProtectedSurface surface="brief-voice">
          <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
            {speaking ? <AiVoiceLabel /> : null}
            {speaking ? <StopButton onStop={stop} /> : <ListenButton disabled={!text.trim() || sessionRunning} onListen={listen} />}
          </div>
        </ProtectedSurface>
        <span role="status" aria-live="polite" className="sr-only" data-brief-voice-status="">
          {speaking ? READING_THE_BRIEF : ''}
        </span>
      </div>
    </VoiceErrorBoundary>
  )
}
