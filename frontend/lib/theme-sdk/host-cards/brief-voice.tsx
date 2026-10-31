/**
 * PRD-268 §6.12.6, K11 (US-C21e) — the SDK's `voice.BriefVoice`: Listen, the host's control that reads the day's
 * brief aloud (components/voice/brief-voice.tsx).
 *
 * Core draws all of it: Listen, the "AI voice assistant" caption and Stop sit in the protected surface `brief-voice`,
 * no theme stylesheet can hide them, and nothing plays until Listen is pressed. The words are the host's too: Listen
 * reads the `spoken` text of the day's brief the host holds (hooks/use-brief.ts: the brief the Auto thread opened),
 * never text a theme hands it, because a theme never speaks (core's review blocks speechSynthesis in a theme's code,
 * services/themes/audio_capture.py). Until the host holds the day's brief, Listen is disabled. A theme places it and
 * follows it: its root (`data-brief-voice="root"`) carries `data-voice-speaking="true"` while Auto reads.
 *
 * SDK 1.4.0: which row Auto is reading. The root carries `data-voice-row="<index>"` while Auto reads one of the brief's
 * rows (its index in the rows of `brief.sections`, in the sections' order: the number Auto says, less one), and the
 * theme's `onSpeakingRow` hears the index, then null between rows and once the reading ends, straight from the
 * speech's own events. Its only prop, checked: a value that isn't a function throws, naming it; a handler that throws
 * is logged, and Auto reads on.
 */
import type { ReactElement } from 'react'

import { BriefVoice as HostBriefVoice } from '@/components/voice/brief-voice'
import { useBrief } from '@/hooks/use-brief'

import { handlerOf } from './proactive-props'

export interface BriefVoiceProps {
  /** Hears the brief row Auto starts reading (its index, in the sections' order), and null once it reads none. */
  readonly onSpeakingRow?: (row: number | null) => void
}

/** Nothing to read: Listen is disabled. */
const NOTHING_SPOKEN = ''
const NO_PROPS: BriefVoiceProps = Object.freeze({})

/** The host's Listen for the day's brief the host holds. */
export function BriefVoice(props: BriefVoiceProps = NO_PROPS): ReactElement {
  const onSpeakingRow = handlerOf(props?.onSpeakingRow, 'BriefVoice', 'onSpeakingRow')
  const brief = useBrief()
  return <HostBriefVoice spoken={brief.data?.spoken ?? NOTHING_SPOKEN} onSpeakingRow={onSpeakingRow} />
}
