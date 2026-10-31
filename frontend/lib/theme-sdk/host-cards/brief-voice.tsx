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
 */
import type { ReactElement } from 'react'

import { BriefVoice as HostBriefVoice } from '@/components/voice/brief-voice'
import { useBrief } from '@/hooks/use-brief'

/** Nothing to read: Listen is disabled. */
const NOTHING_SPOKEN = ''

/** The host's Listen for the day's brief the host holds. */
export function BriefVoice(): ReactElement {
  const brief = useBrief()
  return <HostBriefVoice spoken={brief.data?.spoken ?? NOTHING_SPOKEN} />
}
