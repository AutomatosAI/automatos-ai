'use client'

/**
 * PRD-268 §6.11.3 (US-W05) — Settings → Orchestrator, "What Auto depends on": so a quiet day never hides a broken
 * connection.
 *
 * The server's items come from useReadiness (GET /api/readiness); ReadinessList draws them with their glyph, word and
 * fix, and its Check again posts POST /api/readiness/check ("Checked a moment ago" when the server limits it). After
 * them come this browser's microphone and speech rows, which the server never claims. They come from the browser
 * support useVoice().available is made of (useBrowserVoiceSupport): /settings has no VoiceProvider, so useVoice()
 * would answer nothing available here, and `available.live` also folds in the workspace's plan.
 *
 * - Microphone: ready when this browser can capture audio (what an Auto Live call opens).
 * - Speech: ready when it has speech recognition and synthesis (what hold to talk needs): `available.quick`.
 * - Both are not checked only until the browser is read, after mount.
 *
 * There is no calendar row: no calendar is in v1 (§12-Q20), and neither the server nor this section adds one.
 * /settings is a host-owned route (HOST_OWNED_ROUTES), so the section is the host's under every theme.
 */
import { useMemo } from 'react'
import { ReadinessList } from '@/components/proactive/readiness-list'
import { useReadiness } from '@/hooks/use-readiness'
import type { ReadinessItem, ReadinessState } from '@/lib/proactive/types'
import { type BrowserVoiceSupport, useBrowserVoiceSupport } from '@/lib/voice/host/use-voice-availability'

export const MICROPHONE_LABEL = 'Microphone'
export const SPEECH_LABEL = 'Speech'

/** The browser rows' details: the client's own words, since the server never sees this browser. */
export const BROWSER_DETAILS = Object.freeze({
  checking: 'Still checking this browser.',
  microphoneReady: 'This browser can open the microphone for voice.',
  microphoneMissing:
    "This browser can't open the microphone on this page. Voice needs a current Chrome, Edge or Safari, at an https or localhost address.",
  speechReady: "This browser can hear you and read Auto's replies aloud.",
  speechMissing: "This browser can't recognise speech, so hold to talk won't work here. Try a current Chrome, Edge or Safari.",
})

function browserItem(id: 'microphone' | 'speech', label: string, state: ReadinessState, detail: string): ReadinessItem {
  return { id, label, detail, state, needed_by: ['voice'] }
}

function microphoneItem({ read, capture }: BrowserVoiceSupport): ReadinessItem {
  if (!read) return browserItem('microphone', MICROPHONE_LABEL, 'not_checked', BROWSER_DETAILS.checking)
  return capture
    ? browserItem('microphone', MICROPHONE_LABEL, 'ready', BROWSER_DETAILS.microphoneReady)
    : browserItem('microphone', MICROPHONE_LABEL, 'needs_setup', BROWSER_DETAILS.microphoneMissing)
}

function speechItem({ read, quick }: BrowserVoiceSupport): ReadinessItem {
  if (!read) return browserItem('speech', SPEECH_LABEL, 'not_checked', BROWSER_DETAILS.checking)
  return quick
    ? browserItem('speech', SPEECH_LABEL, 'ready', BROWSER_DETAILS.speechReady)
    : browserItem('speech', SPEECH_LABEL, 'needs_setup', BROWSER_DETAILS.speechMissing)
}

/** This browser's microphone and speech rows; drawn after the server's items. */
export function browserReadinessItems(browser: BrowserVoiceSupport): readonly ReadinessItem[] {
  return [microphoneItem(browser), speechItem(browser)]
}

/** What Auto depends on: the server's items, this browser's microphone and speech, and Check again. */
export function WhatAutoDependsOn() {
  const readiness = useReadiness()
  const browser = useBrowserVoiceSupport()
  const clientItems = useMemo(() => browserReadinessItems(browser), [browser])
  return <ReadinessList readiness={readiness.data} error={readiness.error} clientItems={clientItems} />
}
